# image_utils.py
import cv2
import numpy as np


def extract_traffic_sign_regions(frame, traffic_results, padding=10):
    """提取交通标志区域"""
    crops, bboxes = [], []
    if not traffic_results or len(traffic_results) == 0:
        return crops, bboxes
    
    boxes = traffic_results[0].boxes
    if boxes is None:
        return crops, bboxes
    
    h, w = frame.shape[:2]
    for i, box in enumerate(boxes.xyxy):
        conf = boxes.conf[i] if boxes.conf is not None else 1.0
        if conf < 0.60:
            continue
        
        x1, y1, x2, y2 = map(int, box[:4])
        cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
        cx2, cy2 = min(w, x2 + padding), min(h, y2 + padding)
        
        if cx2 > cx1 and cy2 > cy1:
            crops.append(frame[cy1:cy2, cx1:cx2])
            bboxes.append((x1, y1, x2, y2))
    
    return crops, bboxes


def extract_road_sign_regions(frame, road_results, padding=20):
    """提取路牌区域（大padding，给颜色提取留空间）"""
    regions = []
    if not road_results or len(road_results) == 0:
        return regions
    
    boxes = road_results[0].boxes
    if boxes is None:
        return regions
    
    h, w = frame.shape[:2]
    for i, box in enumerate(boxes.xyxy):
        conf = boxes.conf[i] if boxes.conf is not None else 1.0
        if conf < 0.80:
            continue
        
        x1, y1, x2, y2 = map(int, box[:4])
        cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
        cx2, cy2 = min(w, x2 + padding), min(h, y2 + padding)
        
        if cx2 > cx1 and cy2 > cy1:
            regions.append({
                'region': frame[cy1:cy2, cx1:cx2],
                'bbox': (x1, y1, x2, y2)
            })
    
    return regions


def _order_points(pts):
    """四点排序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_road_sign_precise(image):
    """
    通过颜色对比精确提取路牌矩形区域
    1. HSV颜色分割（蓝/绿路牌）
    2. 找最大轮廓
    3. 四边形透视矫正
    4. 失败则返回原图
    """
    if image is None or image.size == 0:
        return image
    
    h, w = image.shape[:2]
    if h < 30 or w < 30:
        return image
    
    # ===== 新增：归一化增强对比度 =====
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # =================================
    
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 蓝色路牌
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 绿色路牌
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 合并掩码
    mask = cv2.bitwise_or(mask_blue, mask_green)
    
    # 形态学：去噪 + 连接碎片
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # 找最大轮廓
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    # 面积太小则返回原图
    if area < (h * w * 0.08):
        return image
    
    # 多边形近似
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    if len(approx) == 4:
        # 四边形 -> 透视矫正
        pts = approx.reshape(4, 2).astype(np.float32)
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect
        max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 50)
        max_h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 50)
        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_w, max_h))
    else:
        # 非四边形 -> 最小外接矩形裁剪
        x, y, bw, bh = cv2.boundingRect(largest)
        x, y = max(0, x), max(0, y)
        return image[y:y+bh, x:x+bw]


def _simple_iou(a, b):
    """计算两个框的 IoU"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def smooth_detections(history, current_crops, current_bboxes, min_count=3):
    """帧间平滑：目标需在滑动窗口中出现 min_count 次"""
    if len(history) < 3 or not current_bboxes:
        return current_crops, current_bboxes
    
    stable_crops, stable_bboxes = [], []
    for crop, bbox in zip(current_crops, current_bboxes):
        count = sum(
            1 for hist_bboxes in history
            if any(_simple_iou(bbox, hb) > 0.3 for hb in hist_bboxes)
        )
        if count >= min_count:
            stable_crops.append(crop)
            stable_bboxes.append(bbox)
    
    return stable_crops, stable_bboxes