# visualizer.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    """输入原始图片和检测结果，输出画框后的图片"""
    
    COLORS = {
        'traffic': (255, 0, 0),
        'road': (0, 255, 0),
    }
    
    @staticmethod
    def draw_all_results(frame, traffic_results, road_results, ocr_results=None):
        annotated_frame = frame.copy()
        
        if traffic_results and len(traffic_results) > 0:
            annotated_frame = Visualizer._draw_boxes(
                annotated_frame, traffic_results[0],
                color=Visualizer.COLORS['traffic'], label_prefix='TS: ', min_conf=0.70
            )
        
        if road_results and len(road_results) > 0:
            annotated_frame = Visualizer._draw_boxes(
                annotated_frame, road_results[0],
                color=Visualizer.COLORS['road'], label_prefix='RS: ', min_conf=0.80
            )
        
        if ocr_results:
            for item in ocr_results:
                box = item[0]
                text = item[1][0]
                conf = item[1][1]
                pts = np.array(box, dtype=np.int32)
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                x, y = int(pts[0][0]), int(pts[0][1]) - 5
                display_text = f"OCR: {text} ({conf:.2f})"
                draw_chinese_text(annotated_frame, display_text, (x, y), font_size=18, color=(0, 255, 255))
        
        return annotated_frame
    
    @staticmethod
    def _draw_boxes(frame, results, color, label_prefix='', min_conf=0.0):
        boxes = results.boxes
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                conf = boxes.conf[i] if boxes.conf is not None else 1.0
                if conf < min_conf:
                    continue
                cls = int(boxes.cls[i]) if boxes.cls is not None else 0
                name = results.names[cls] if hasattr(results, 'names') else f'Class_{cls}'
                
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{label_prefix}{name} {conf:.2f}"
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                
                if any('\u4e00' <= char <= '\u9fff' for char in label):
                    draw_chinese_text(frame, label, (x1, text_y), font_size=14, color=color)
                else:
                    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame


def draw_chinese_text(img, text, position, font_size=20, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)