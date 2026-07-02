# display_window.py
# pyright: reportDeprecated=false, reportArgumentType=false
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
                             QFrame, QSizePolicy, QSplitter, QPushButton)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import queue
import time


class VideoDisplayWindow(QMainWindow):
    """PyQt 视频显示窗口 - 左侧视频，右侧检测结果"""
    
    pause_toggled = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志检测系统 - 实时监控")
        self.setMinimumSize(1600, 900)
        self.display_queue = queue.Queue(maxsize=10)
        self.traffic_targets = {}
        self.road_targets = {}
        self.next_traffic_id = 0
        self.next_road_id = 0
        self.display_timeout = 1.0
        self.iou_threshold = 0.3
        self.paused = False
        
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.start(33)
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.cleanup_expired_targets)
        self.cleanup_timer.start(200)
        self.center_window()
        self.show()
    
    def center_window(self):
        screen = QApplication.primaryScreen().geometry() # type: ignore
        self.move((screen.width() - self.width()) // 2,
                 (screen.height() - self.height()) // 2)
    
    def _calculate_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _match_target(self, bbox, targets, target_type=''):
        best_iou = 0
        best_id = None
        threshold = 0.1 if target_type == 'traffic' else self.iou_threshold
        for target_id, target_info in targets.items():
            iou = self._calculate_iou(bbox, target_info['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_id = target_id
        return best_id if best_iou > threshold else None
    
    def _update_targets(self, crops, bboxes, targets, target_type):
        current_time = time.time()
        matched_ids = set()
        if len(crops) != len(bboxes):
            return []
        
        alive_count = sum(1 for t in targets.values() 
                         if current_time - t['last_seen'] < self.display_timeout or self.paused)
        max_new = max(0, len(crops) - alive_count)
        new_added = 0
        
        for crop, bbox in zip(crops, bboxes):
            if crop is None or crop.size == 0:
                continue
            current_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            matched_id = self._match_target(bbox, targets, target_type)
            if matched_id is not None:
                old_bbox = targets[matched_id]['bbox']
                old_area = (old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])
                if current_area >= old_area:
                    targets[matched_id]['crop'] = crop
                    targets[matched_id]['bbox'] = bbox
                if not self.paused:
                    targets[matched_id]['last_seen'] = current_time
                matched_ids.add(matched_id)
            else:
                if new_added < max_new:
                    if target_type == 'traffic':
                        new_id = self.next_traffic_id
                        self.next_traffic_id += 1
                    else:
                        new_id = self.next_road_id
                        self.next_road_id += 1
                    targets[new_id] = {'crop': crop, 'bbox': bbox, 'last_seen': current_time}
                    matched_ids.add(new_id)
                    new_added += 1
        
        return [targets[tid]['crop'] for tid in sorted(targets.keys()) 
                if current_time - targets[tid]['last_seen'] < self.display_timeout or self.paused]
    
    def cleanup_expired_targets(self):
        if self.paused:
            return
        current_time = time.time()
        et = [tid for tid, t in self.traffic_targets.items() if current_time - t['last_seen'] >= self.display_timeout]
        er = [rid for rid, t in self.road_targets.items() if current_time - t['last_seen'] >= self.display_timeout]
        for tid in et: del self.traffic_targets[tid]
        for rid in er: del self.road_targets[rid]
        if et or er: self.refresh_display()
    
    def refresh_display(self):
        current_time = time.time()
        tc = [self.traffic_targets[tid]['crop'] for tid in sorted(self.traffic_targets.keys()) if current_time - self.traffic_targets[tid]['last_seen'] < self.display_timeout]
        rc = [self.road_targets[rid]['crop'] for rid in sorted(self.road_targets.keys()) if current_time - self.road_targets[rid]['last_seen'] < self.display_timeout]
        self.update_crops_grid(tc, self.traffic_grid, "未检测到交通标志")
        self.update_crops_grid(rc, self.road_grid, "未检测到路牌")
    
    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.setText("▶ 继续")
            self.btn_pause.setStyleSheet(self._pause_style("#4CAF50", "#fff"))
        else:
            self.btn_pause.setText("⏸ 暂停")
            self.btn_pause.setStyleSheet(self._pause_style("#FF5722", "#fff"))
        self.pause_toggled.emit()
    
    def _pause_style(self, bg, fg):
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
        """
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        splitter = QSplitter(Qt.Horizontal)# type: ignore
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        title_widget = QWidget()
        title_widget.setStyleSheet("QWidget { background-color: #1976D2; border-radius: 8px; padding: 6px; }")
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(10, 4, 10, 4)
        
        title_label = QLabel("📹 实时视频流")
        title_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        self.btn_record = QPushButton("🔴 录制")
        self.btn_record.setFixedSize(120, 40)
        self.btn_record.setStyleSheet(self._pause_style("#9E9E9E", "#fff"))
        title_layout.addWidget(self.btn_record)
        
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setFixedSize(120, 40)
        self.btn_pause.setStyleSheet(self._pause_style("#FF5722", "#fff"))
        self.btn_pause.clicked.connect(self.toggle_pause)
        title_layout.addWidget(self.btn_pause)
        
        left_layout.addWidget(title_widget)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(1280, 720)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("QLabel { border: 3px solid #1976D2; border-radius: 10px; background-color: #000; }")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label, stretch=1)
        
        self.info_label = QLabel("等待视频...")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #666; padding: 5px; font-size: 12px;")
        left_layout.addWidget(self.info_label)
        
        splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(15)
        right_title = QLabel("🔍 检测结果")
        right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_title.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; color: #333; background-color: #E3F2FD; border-radius: 8px; padding: 10px; }")
        right_layout.addWidget(right_title)
        traffic_widget, self.traffic_grid = self.create_detection_section("🚗 交通标志 Traffic Signs", "#FF5722", "未检测到交通标志")
        right_layout.addWidget(traffic_widget, stretch=35)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("QFrame { color: #BDBDBD; max-height: 2px; }")
        right_layout.addWidget(separator)
        road_ocr_widget = QWidget()
        road_ocr_layout = QVBoxLayout(road_ocr_widget)
        road_ocr_layout.setSpacing(10)
        road_widget, self.road_grid = self.create_detection_section("🛣️ 路牌标志 Road Signs", "#4CAF50", "未检测到路牌")
        road_ocr_layout.addWidget(road_widget, stretch=60)
        ocr_widget = self.create_ocr_section()
        road_ocr_layout.addWidget(ocr_widget, stretch=40)
        right_layout.addWidget(road_ocr_widget, stretch=65)
        scroll = QScrollArea()
        scroll.setWidget(right_widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        splitter.addWidget(scroll)
        splitter.setSizes([1280, 720])
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(splitter)
        self.setStyleSheet("QMainWindow { background-color: #F5F5F5; }")
    
    def create_detection_section(self, title, color, placeholder_text):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        section_title = QLabel(title)
        section_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        section_title.setStyleSheet(f"QLabel {{ font-size: 14px; font-weight: bold; color: white; background-color: {color}; border-radius: 6px; padding: 8px; }}")
        layout.addWidget(section_title)
        grid_container = QWidget()
        grid_container.setStyleSheet("QWidget { background-color: white; border: 2px solid #E0E0E0; border-radius: 8px; min-height: 120px; }")
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(10)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(grid_container)
        return widget, grid_layout
    
    def create_ocr_section(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        title = QLabel("📝 OCR 识别文字")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: white; background-color: #9C27B0; border-radius: 6px; padding: 8px; }")
        layout.addWidget(title)
        self.ocr_text = QLabel("等待OCR识别结果...")
        self.ocr_text.setWordWrap(True)
        self.ocr_text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.ocr_text.setTextFormat(Qt.PlainText)# type: ignore
        self.ocr_text.setStyleSheet("QLabel { background-color: white; border: 2px solid #E0E0E0; border-radius: 8px; padding: 15px; font-size: 14px; color: #333; min-height: 120px; }")
        layout.addWidget(self.ocr_text)
        return widget
    
    def check_queue(self):
        try:
            if not self.display_queue.empty():
                data = self.display_queue.get_nowait()
                self.update_all(*data)
        except queue.Empty: pass
        except Exception as e: print(f"队列错误: {e}")
    
    def add_frame(self, frame, traffic_crops, traffic_bboxes, road_crops, road_bboxes, ocr_text, frame_count=None):
        try:
            if self.display_queue.full():
                try: self.display_queue.get_nowait()
                except: pass
            self.display_queue.put_nowait((frame, traffic_crops, traffic_bboxes, road_crops, road_bboxes, ocr_text, frame_count))
        except Exception as e: print(f"添加帧错误: {e}")
    
    def update_all(self, frame, traffic_crops, traffic_bboxes, road_crops, road_bboxes, ocr_text, frame_count):
        if frame is not None: self.update_video(frame, frame_count)
        dt = self._update_targets(traffic_crops if traffic_crops else [], traffic_bboxes if traffic_bboxes else [], self.traffic_targets, 'traffic')
        dr = self._update_targets(road_crops if road_crops else [], road_bboxes if road_bboxes else [], self.road_targets, 'road')
        self.update_crops_grid(dt, self.traffic_grid, "未检测到交通标志")
        self.update_crops_grid(dr, self.road_grid, "未检测到路牌")
        if ocr_text is not None: self.update_ocr_text(ocr_text)
    
    def update_video(self, frame, frame_count=None):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            tw = 1280
            th = int(h * (tw / w))
            rgb = cv2.resize(rgb, (tw, th))
            qimg = QImage(rgb.data, tw, th, 3 * tw, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(pix)
            if frame_count is not None: self.info_label.setText(f"帧: {frame_count} | {w}x{h} | 显示: {tw}x{th}")
        except Exception as e: print(f"视频错误: {e}")
    
    def update_crops_grid(self, crops, grid_layout, empty_text="无检测结果"):
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if not crops:
            placeholder = QLabel(empty_text)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #999; font-style: italic; font-size: 13px; padding: 30px;")
            grid_layout.addWidget(placeholder, 0, 0, Qt.AlignmentFlag.AlignCenter)
            return
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0: continue
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                s = 110 / max(h, w)
                nw, nh = max(40, int(w * s)), max(40, int(h * s))
                rgb = cv2.resize(rgb, (nw, nh))
                qimg = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888)
                lbl = QLabel()
                lbl.setPixmap(QPixmap.fromImage(qimg))
                lbl.setFixedSize(nw + 10, nh + 10)
                lbl.setStyleSheet("border:2px solid #BDBDBD; border-radius:6px; background:#fff;")
                grid_layout.addWidget(lbl, 0, i)
            except: continue
        grid_layout.setColumnStretch(grid_layout.columnCount(), 1)
    
    def update_ocr_text(self, text):
        if text:
            self.ocr_text.setText(text)
            self.ocr_text.setStyleSheet("QLabel { background-color: #FFFDE7; border: 2px solid #9C27B0; border-radius: 8px; padding: 15px; font-size: 14px; color: #333; min-height: 120px; }")
        else:
            self.ocr_text.setText("未识别到文字")
            self.ocr_text.setStyleSheet("QLabel { background-color: white; border: 2px solid #E0E0E0; border-radius: 8px; padding: 15px; font-size: 14px; color: #999; min-height: 120px; }")
    
    def closeEvent(self, event):# type: ignore
        self.timer.stop()
        self.cleanup_timer.stop()
        print("✅ 显示窗口已关闭")
        event.accept()