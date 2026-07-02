# main.py
import sys
import os

# for Windows, add torch lib path to PATH to avoid DLL load errors when encountering "ImportError: DLL load failed while importing _C: 找不到指定的模块" 
# torch_lib = r".\Environment\Python\data\Lib\site-packages\torch\lib"
# if os.path.exists(torch_lib):
#     os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
#     if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
#         try: os.add_dll_directory(torch_lib)
#         except: pass
        
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from detector import Detector
from visualizer import Visualizer
from ocr_detector import PaddleOCRDetector
from nms_utils import NMSUtils
from display_window import VideoDisplayWindow
from image_utils import (extract_traffic_sign_regions, extract_road_sign_regions,
                          smooth_detections, extract_road_sign_precise)


def perform_ocr_on_regions(ocr_detector, regions):
    all_ocr_results = []
    for ri in regions:
        region, (x1, y1, x2, y2) = ri['region'], ri['bbox']
        result = ocr_detector.predict(region, save_path=None, show_result=False)
        if result:
            for item in result:
                box, (text, conf) = item[0], item[1]
                all_ocr_results.append([[[p[0] + x1, p[1] + y1] for p in box], (text, conf)])
    if all_ocr_results:
        return NMSUtils.ocr_nms(all_ocr_results, iou_threshold=0.3, conf_threshold=0.50, merge_threshold=0.7)
    return []


def format_ocr_text(ocr_results, road_bboxes):
    if not ocr_results: return ""
    road_texts = {i: [] for i in range(len(road_bboxes))}
    unmatched = []
    for item in ocr_results:
        box, (text, conf) = item[0], item[1]
        if conf < 0.50: continue
        cx = sum(p[0] for p in box) / 4
        cy = sum(p[1] for p in box) / 4
        matched = False
        for i, bbox in enumerate(road_bboxes):
            if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                if text not in road_texts[i]: road_texts[i].append(text)
                matched = True; break
        if not matched: unmatched.append(text)
    lines = [f"路牌{i+1}: {' '.join(t)}" for i, t in road_texts.items() if t]
    if unmatched: lines.append("其他: " + " ".join(unmatched))
    return "\n\n".join(lines)


def capture_window(window):
    pixmap = window.grab()
    qimg = pixmap.toImage()
    qimg = qimg.convertToFormat(QImage.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.array(ptr).reshape(h, w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def process_video(display_window, video_path):
    print("初始化检测器...")
    traffic_detector = Detector("trafficSign.pt")
    road_detector = Detector("roadSign.pt")
    ocr_detector = PaddleOCRDetector(lang='ch')
    
    import torch
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "⚠️ CPU模式")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("❌ 无法打开视频"); return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"📹 {w}x{h} @ {fps}fps")
    
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter.fourcc(*'XVID'), fps, (w, h))
    
    recording = False
    record_writer = None
    
    def start_recording():
        nonlocal recording, record_writer
        filename, _ = QFileDialog.getSaveFileName(display_window, "保存录制文件", "screen_record.avi", "AVI (*.avi)")
        if filename:
            sample = capture_window(display_window)
            rh, rw = sample.shape[:2]
            record_writer = cv2.VideoWriter(filename, cv2.VideoWriter.fourcc(*'XVID'), 24, (rw, rh))
            recording = True
            display_window.btn_record.setText("⏹ 停止录制")
            display_window.btn_record.setStyleSheet(display_window._pause_style("#f44336", "#fff"))
            print(f"🔴 开始录制界面: {filename} ({rw}x{rh})")
    
    def stop_recording():
        nonlocal recording, record_writer
        recording = False
        if record_writer:
            record_writer.release()
            record_writer = None
        display_window.btn_record.setText("🔴 录制")
        display_window.btn_record.setStyleSheet(display_window._pause_style("#9E9E9E", "#fff"))
        print("⏹ 录制已停止")
    
    def toggle_recording():
        if recording:
            stop_recording()
        else:
            start_recording()
    
    # 绑定录制按钮（按钮在 display_window 里创建）
    display_window.btn_record.clicked.connect(toggle_recording)
    
    print("🚀 开始 | 按钮/空格=暂停 录制=录制整个界面 Q=退出")
    
    frame_count, ocr_interval = 0, 5
    last_ocr, last_bboxes = None, []
    paused = False
    frame_time, last_time = 1.0 / fps, time.time()
    road_history = []
    
    def on_pause():
        nonlocal paused
        paused = not paused
        print("⏸️ 暂停" if paused else "▶️ 继续")
    
    display_window.pause_toggled.connect(on_pause)
    
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == 32:
            display_window.toggle_pause()
        
        if display_window.paused:
            QApplication.processEvents()
            continue
        
        ret, frame = cap.read()
        if not ret: break
        
        traffic_results = traffic_detector.predict(frame)
        road_results = road_detector.predict(frame)
        
        traffic_crops, traffic_bboxes = extract_traffic_sign_regions(frame, traffic_results)
        road_regions = extract_road_sign_regions(frame, road_results)
        
        for r in road_regions:
            r['region'] = extract_road_sign_precise(r['region'])
        
        road_crops = [r['region'] for r in road_regions]
        road_bboxes = [r['bbox'] for r in road_regions]
        
        road_history.append(road_bboxes)
        if len(road_history) > 5: road_history.pop(0)
        road_crops, road_bboxes = smooth_detections(road_history, road_crops, road_bboxes, 3)
        
        ocr_text = ""
        if frame_count % ocr_interval == 0 and road_regions:
            cur = perform_ocr_on_regions(ocr_detector, road_regions)
            if cur: last_ocr, last_bboxes = cur, road_bboxes
        if last_ocr: ocr_text = format_ocr_text(last_ocr, last_bboxes)
        
        annotated = frame.copy()
        if traffic_results and len(traffic_results) > 0:
            annotated = Visualizer._draw_boxes(annotated, traffic_results[0], color=Visualizer.COLORS['traffic'], label_prefix='TS: ', min_conf=0.70)
        if road_results and len(road_results) > 0:
            annotated = Visualizer._draw_boxes(annotated, road_results[0], color=Visualizer.COLORS['road'], label_prefix='RS: ', min_conf=0.80)
        
        display_window.add_frame(annotated, traffic_crops, traffic_bboxes, road_crops, road_bboxes, ocr_text, frame_count)
        out.write(annotated)
        
        if recording and record_writer:
            screen_frame = capture_window(display_window)
            record_writer.write(screen_frame)
        
        frame_count += 1
        
        elapsed = time.time() - last_time
        if elapsed < frame_time: time.sleep(frame_time - elapsed)
        last_time = time.time()
        QApplication.processEvents()
    
    if recording:
        stop_recording()
    cap.release(); out.release()
    print(f"✅ 完成！共{frame_count}帧")


def main():
    app = QApplication(sys.argv)
    window = VideoDisplayWindow()
    video_path = "xxxxxx" # Default video path ,change it to your own video path
    QTimer.singleShot(100, lambda: process_video(window, video_path = video_path))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()