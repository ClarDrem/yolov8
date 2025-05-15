import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from yolo import YOLO
import numpy as np
import os
from PIL import Image


class YOLOConfusionMatrix:
    def __init__(self, yolo_model, test_annotation_path, num_classes):
        self.yolo = yolo_model
        self.num_classes = num_classes
        self.test_annotation_path = test_annotation_path
        self.confusion = np.zeros((num_classes + 1, num_classes + 1))  # +1 for background/false positives

    def _parse_annotations(self, annotation_path):
        """
        解析标注文件，假设标注文件格式为：
        <class_id> <x_center> <y_center> <width> <height>
        """
        with open(annotation_path) as f:
            lines = f.readlines()
        true_boxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            box = list(map(float, parts[1:5]))
            true_boxes.append((class_id, box))
        return true_boxes

    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IOU（交并比）
        """
        # 转换格式为x1,y1,x2,y2
        box1 = self._xywh2xyxy(box1)
        box2 = self._xywh2xyxy(box2)

        # 计算交集区域
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def _xywh2xyxy(self, x):
        """
        转换坐标格式从中心坐标(x_center, y_center, width, height)
        到角点坐标(x1, y1, x2, y2)
        """
        x_center, y_center, w, h = x
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return [x1, y1, x2, y2]

    def evaluate(self, iou_threshold=0.5):
        """
        遍历测试集生成混淆矩阵
        """
        # 获取测试集文件列表
        test_files = [f for f in os.listdir(self.test_annotation_path) if f.endswith('.txt')]

        for file_name in test_files:
            image_id = os.path.splitext(file_name)[0]
            image_path = os.path.join("你的图像目录", image_id + ".jpg")  # 需修改为实际路径

            # 读取图像和标注
            image = Image.open(image_path)
            true_boxes = self._parse_annotations(os.path.join(self.test_annotation_path, file_name))

            # 获取预测结果
            self.yolo.detect_image(image, count=False)
            pred_boxes = self.yolo.top_boxes  # 需要添加这些属性记录
            pred_classes = self.yolo.top_label
            pred_scores = self.yolo.top_conf

            # 匹配预测和真实框
            matched_true = set()
            for pred_idx, (pred_class, pred_box) in enumerate(zip(pred_classes, pred_boxes)):
                best_iou = 0
                best_true_idx = -1

                for true_idx, (true_class, true_box) in enumerate(true_boxes):
                    iou = self._calculate_iou(pred_box, true_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_true_idx = true_idx

                if best_true_idx != -1:
                    true_class = true_boxes[best_true_idx][0]
                    if best_true_idx not in matched_true:
                        self.confusion[true_class][pred_class] += 1
                        matched_true.add(best_true_idx)
                    else:
                        self.confusion[self.num_classes][pred_class] += 1  # FP（重复检测）
                else:
                    self.confusion[self.num_classes][pred_class] += 1  # FP（误检）

            # 处理未匹配的真实框（FN）
            for true_idx, (true_class, true_box) in enumerate(true_boxes):
                if true_idx not in matched_true:
                    self.confusion[true_class][self.num_classes] += 1

    def plot(self, class_names):
        """
        绘制混淆矩阵
        """
        plt.figure(figsize=(15, 12))
        sns.heatmap(self.confusion, annot=True, fmt='g',
                    xticklabels=class_names + ['FN'],
                    yticklabels=class_names + ['FP'],
                    cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


# 使用示例
if __name__ == "__main__":
    yolo = YOLO()
    cm = YOLOConfusionMatrix(yolo,
                             test_annotation_path="你的标注目录",  # 修改为实际路径
                             num_classes=yolo.num_classes)
    cm.evaluate()
    cm.plot(yolo.class_names)