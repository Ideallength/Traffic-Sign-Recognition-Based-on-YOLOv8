import cv2
from ultralytics import YOLO

class Detector:
    """输入原始图片，输出 YOLO 检测结果"""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.model_name = model_path
        print(f"Detector initialized with model: {model_path}")

    def predict(self, frame):
        results = self.model(frame)
        return results

    def get_name(self):
        return self.model_name