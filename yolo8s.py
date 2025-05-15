# --------------------------------------#
#       对数据集进行训练
# --------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

np.object = object
if __name__ == "__main__":
    # -------------------------------#
    #   硬件配置
    # -------------------------------#
    Cuda = True
    seed = 42  # 使用更通用的随机种子
    distributed = False
    sync_bn = False
    fp16 = False  # 保持混合精度训练

    # -------------------------------#
    #   模型与数据配置
    # -------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    model_path = 'model_data/yolov8_m.pth'
    input_shape = [448, 448
                   ]  # 恢复YOLO标准输入尺寸（显存不足可降为448）
    phi = 'm'  # 中等尺寸模型
    pretrained = True  # 必须启用预训练

    # -------------------------------#
    #   数据增强配置
    # -------------------------------#
    mosaic = True
    mosaic_prob = 0.5  # 适当降低mosaic概率
    mixup = True
    mixup_prob = 0.3  # 降低mixup概率
    special_aug_ratio = 0.6  # 特殊数据增强比例

    # -------------------------------#
    #   训练策略配置
    # -------------------------------#
    label_smoothing = 0.1  # 启用标签平滑
    Init_Epoch = 0
    Freeze_Epoch = 15  # 冻结训练阶段
    Freeze_batch_size = 32  # 冻结阶段批次大小
    UnFreeze_Epoch = 100  # 总训练轮次增加
    Unfreeze_batch_size = 16
    Freeze_Train = True

    # -------------------------------#
    #   优化器配置
    # -------------------------------#
    Init_lr = 1e-2  # Adam优化器推荐初始学习率
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 5e-4  # 增强权重衰减
    lr_decay_type = "cos"

    # -------------------------------#
    #   系统配置
    # -------------------------------#
    save_period = 10
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5  # 每5个epoch验证一次
    num_workers = 8  # 提高数据加载效率
    train_annotation_path = '2025_train.txt'
    val_annotation_path = '2025_val.txt'

    # -------------------------------#
    #   初始化设置
    # -------------------------------#
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()

    # 分布式训练设置（保持默认）
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # -------------------------------#
    #   获取类别信息
    # -------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # -------------------------------#
    #   加载预训练权重
    # -------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)

    # -------------------------------#
    #   模型初始化
    # -------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    # 加载自定义权重（如有）
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # -------------------------------#
    #   损失函数
    # -------------------------------#
    yolo_loss = Loss(model)

    # -------------------------------#
    #   日志记录
    # -------------------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # -------------------------------#
    #   混合精度训练
    # -------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    # -------------------------------#
    #   模型训练模式设置
    # -------------------------------#
    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    # -------------------------------#
    #   多卡并行设置
    # -------------------------------#
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    ema = ModelEMA(model_train)

    # -------------------------------#
    #   数据集加载
    # -------------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # -------------------------------#
    #   训练信息展示
    # -------------------------------#
    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch

    # -------------------------------#
    #   训练阶段设置
    # -------------------------------#
    if True:
        UnFreeze_flag = False

        # 冻结训练阶段
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # 动态批次设置
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # 参数分组
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        # 优化器设置
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        # 学习率调度器
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 数据加载器
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if ema:
            ema.updates = epoch_step * Init_Epoch

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                    mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
                                    train=True, special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                  mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                  special_aug_ratio=0)

        # 分布式采样设置
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # 验证回调
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # -------------------------------#
        #   开始训练
        # -------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 解冻阶段设置
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                # 解冻backbone
                for param in model.backbone.parameters():
                    param.requires_grad = True

                # 重置数据加载器
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if ema:
                    ema.updates = epoch_step * epoch

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            # 设置当前epoch
            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            # 分布式训练设置
            if distributed:
                train_sampler.set_epoch(epoch)

            # 调整学习率
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 训练单个epoch
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir,
                          local_rank)

            # 分布式同步
            if distributed:
                dist.barrier()

        # 关闭日志写入器
        if local_rank == 0:
            loss_history.writer.close()