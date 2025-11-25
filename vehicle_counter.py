import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class VehicleCounter:
    def __init__(self, model_path='yolo11n.pt', line_position=0.5):
        """
        Khởi tạo hệ thống đếm phương tiện
        
        Args:
            model_path: Đường dẫn đến file model YOLOv11
            line_position: Vị trí đường đếm (0.0-1.0, tính từ trên xuống)
        """
        self.model = YOLO(model_path)
        self.line_position = line_position  # Vị trí đường đếm (tỷ lệ chiều cao)
        self.tracks = defaultdict(dict)  # Lưu trữ tracking info
        self.count_up = 0  # Đếm phương tiện đi lên
        self.count_down = 0  # Đếm phương tiện đi xuống
        self.last_update = {}  # Thời gian cập nhật cuối cùng của mỗi ID
        self.vehicle_classes = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
        self.class_names = {
            2: 'Car',
            3: 'Motorcycle', 
            5: 'Bus',
            7: 'Truck'
        }
        
    def update_counts(self, results, frame_height):
        """Cập nhật số lượng phương tiện đã vượt qua đường đếm"""
        current_time = time.time()
        line_y = int(frame_height * self.line_position)
        
        # Lấy các detections có tracking ID
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, ids, classes, confidences):
                if cls not in self.vehicle_classes:
                    continue
                    
                x1, y1, x2, y2 = box
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2
                
                # Khởi tạo tracking cho ID mới
                if track_id not in self.tracks:
                    self.tracks[track_id] = {
                        'center_y': center_y,
                        'center_x': center_x,
                        'crossed': False,
                        'direction': None,
                        'last_y': center_y,
                        'class': cls,
                        'confidence': conf
                    }
                    self.last_update[track_id] = current_time
                else:
                    # Cập nhật vị trí
                    last_y = self.tracks[track_id]['last_y']
                    self.tracks[track_id]['last_y'] = center_y
                    self.tracks[track_id]['center_x'] = center_x
                    self.tracks[track_id]['center_y'] = center_y
                    self.last_update[track_id] = current_time
                    
                    # Xác định hướng di chuyển
                    if self.tracks[track_id]['direction'] is None:
                        if center_y > last_y:
                            self.tracks[track_id]['direction'] = 'down'
                        elif center_y < last_y:
                            self.tracks[track_id]['direction'] = 'up'
                    
                    # Kiểm tra nếu phương tiện vượt qua đường đếm
                    if not self.tracks[track_id]['crossed']:
                        if self.tracks[track_id]['direction'] == 'down':
                            if last_y <= line_y and center_y > line_y:
                                self.count_down += 1
                                self.tracks[track_id]['crossed'] = True
                        elif self.tracks[track_id]['direction'] == 'up':
                            if last_y >= line_y and center_y < line_y:
                                self.count_up += 1
                                self.tracks[track_id]['crossed'] = True
        
        # Xóa các track cũ không được cập nhật trong 2 giây
        tracks_to_remove = []
        for track_id, last_time in self.last_update.items():
            if current_time - last_time > 2.0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]
            if track_id in self.last_update:
                del self.last_update[track_id]
    
    def draw_results(self, frame, results):
        """Vẽ kết quả lên frame"""
        frame_height, frame_width = frame.shape[:2]
        line_y = int(frame_height * self.line_position)
        
        # Vẽ đường đếm
        cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 255), 3)
        cv2.putText(frame, 'Counting Line', (10, line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Vẽ các bounding boxes và thông tin
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, ids, classes, confidences):
                if cls not in self.vehicle_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Màu sắc theo loại phương tiện
                colors = {
                    2: (255, 0, 0),      # Car - Blue
                    3: (0, 255, 0),      # Motorcycle - Green
                    5: (255, 0, 255),    # Bus - Magenta
                    7: (0, 165, 255)     # Truck - Orange
                }
                color = colors.get(cls, (255, 255, 255))
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ label
                label = f"{self.class_names[cls]} {track_id} {conf:.2f}"
                if track_id in self.tracks:
                    direction = self.tracks[track_id]['direction']
                    if direction:
                        label += f" ({direction})"
                
                # Background cho text
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_height - 5), 
                            (x1 + label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hiển thị số lượng đếm được
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.putText(frame, f'Vehicles Up: {self.count_up}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Vehicles Down: {self.count_down}', (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f'Total: {self.count_up + self.count_down}', (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Xử lý một frame"""
        # Chạy YOLOv11 với tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                   classes=self.vehicle_classes, conf=0.25)
        
        # Cập nhật số lượng
        self.update_counts(results, frame.shape[0])
        
        # Vẽ kết quả
        annotated_frame = self.draw_results(frame, results)
        
        return annotated_frame
    
    def reset_counts(self):
        """Reset bộ đếm"""
        self.count_up = 0
        self.count_down = 0
        self.tracks.clear()
        self.last_update.clear()

