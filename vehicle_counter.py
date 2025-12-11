import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import torch

class VehicleCounter:
    def __init__(self, model_path='models/yolo11n.pt', line_position=0.7, 
                 inference_size=640, use_half_precision=True,
                 inference_stride=1):
        """
        Khởi tạo hệ thống đếm phương tiện
        
        Args:
            model_path: Đường dẫn đến file model đã được huấn luyện
            line_position: Vị trí đường đếm (0.0-1.0, tính từ trên xuống)
            inference_size: Kích thước frame để inference (nhỏ hơn = nhanh hơn, mặc định 640)
            use_half_precision: Sử dụng FP16 nếu GPU có sẵn (nhanh hơn ~2x)
            inference_stride: Chỉ chạy inference mỗi N frame (CPU nên >1 để nhẹ hơn)
        """
        self.model = YOLO(model_path)
        
        # Tối ưu hóa model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # CPU không dùng FP16 để tránh overhead chuyển kiểu
        self.use_half = use_half_precision and device == 'cuda'
        
        # Thiết lập inference size (giảm để tăng tốc)
        self.inference_size = inference_size
        # Giảm tần suất inference để nhẹ CPU
        self.inference_stride = max(1, int(inference_stride))
        
        self.line_position = line_position  # Vị trí đường đếm (tỷ lệ chiều cao)
        self.tracks = defaultdict(dict)  # Lưu trữ tracking info
        self.count_up = 0  # Đếm phương tiện đi lên
        self.count_down = 0  # Đếm phương tiện đi xuống
        self.last_update = {}  # Thời gian cập nhật cuối cùng của mỗi ID
        self.vehicle_classes = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
        self.class_names = {
            2: 'Car',
            3: 'Motorbike',
            5: 'Bus',
            7: 'Truck'
        }
        # Lưu số lượng theo từng loại xe và chiều di chuyển
        self.class_counts = {
            cls: {'up': 0, 'down': 0}
            for cls in self.vehicle_classes
        }
        
    def update_counts(self, results, frame_height, scale_x=1.0, scale_y=1.0):
        """Cập nhật số lượng phương tiện đã vượt qua đường đếm"""
        current_time = time.time()
        # QUAN TRỌNG: line_y phải tính theo frame_height gốc (không scale)
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
                # Scale boxes về kích thước gốc nếu đã resize
                # QUAN TRỌNG: Luôn scale về kích thước gốc để so sánh với line_y
                x1 = float(x1 * scale_x)
                x2 = float(x2 * scale_x)
                y1 = float(y1 * scale_y)
                y2 = float(y2 * scale_y)
                
                center_y = (y1 + y2) / 2.0
                center_x = (x1 + x2) / 2.0
                
                # Khởi tạo tracking cho ID mới
                if track_id not in self.tracks:
                    # Lưu center_y đã được scale về kích thước gốc
                    self.tracks[track_id] = {
                        'center_y': center_y,
                        'center_x': center_x,
                        'crossed': False,
                        'direction': None,
                        'last_y': center_y,  # Lưu đã scale về kích thước gốc
                        'class': cls,
                        'confidence': conf
                    }
                    self.last_update[track_id] = current_time
                else:
                    # Cập nhật vị trí
                    # last_y đã được scale về kích thước gốc từ frame trước
                    last_y = self.tracks[track_id]['last_y']
                    self.tracks[track_id]['last_y'] = center_y  # Lưu đã scale về kích thước gốc
                    self.tracks[track_id]['center_x'] = center_x
                    self.tracks[track_id]['center_y'] = center_y
                    self.last_update[track_id] = current_time
                    
                    # Xác định hướng di chuyển
                    if self.tracks[track_id]['direction'] is None:
                        if abs(center_y - last_y) > 1.0:  # Chỉ xác định khi có di chuyển đáng kể
                            if center_y > last_y:
                                self.tracks[track_id]['direction'] = 'down'
                            elif center_y < last_y:
                                self.tracks[track_id]['direction'] = 'up'
                    
                    # Kiểm tra nếu phương tiện vượt qua đường đếm
                    # Tất cả đều ở kích thước gốc nên so sánh chính xác
                    if not self.tracks[track_id]['crossed'] and self.tracks[track_id]['direction'] is not None:
                        if self.tracks[track_id]['direction'] == 'down':
                            # Đi xuống: từ trên line_y xuống dưới line_y
                            # Cho phép một khoảng tolerance để tránh bỏ sót
                            if (last_y < line_y + 10) and center_y > line_y - 10:
                                self.count_down += 1
                                self.class_counts[cls]['down'] += 1
                                self.tracks[track_id]['crossed'] = True
                        elif self.tracks[track_id]['direction'] == 'up':
                            # Đi lên: từ dưới line_y lên trên line_y
                            # Cho phép một khoảng tolerance để tránh bỏ sót
                            if (last_y > line_y - 10) and center_y < line_y + 10:
                                self.count_up += 1
                                self.class_counts[cls]['up'] += 1
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
    
    def draw_results(self, frame, results, scale_x=1.0, scale_y=1.0):
        """Vẽ kết quả lên frame"""
        frame_height, frame_width = frame.shape[:2]
        line_y = int(frame_height * self.line_position)
        
        # Vẽ đường đếm
        cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 255), 3)
        cv2.putText(frame, 'Counting Line', (10, line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Vẽ các bounding boxes và thông tin
        if results is not None and len(results) > 0 and results[0].boxes.id is not None:
            # Chuyển sang numpy một lần duy nhất
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Scale boxes về kích thước gốc
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            # Lọc chỉ các vehicle classes trước khi vẽ
            vehicle_mask = np.isin(classes, self.vehicle_classes)
            boxes = boxes[vehicle_mask]
            ids = ids[vehicle_mask]
            classes = classes[vehicle_mask]
            confidences = confidences[vehicle_mask]
            
            # Màu sắc theo loại phương tiện (định nghĩa một lần)
            colors = {
                2: (255, 0, 0),      # Car - Blue
                3: (0, 255, 0),      # Motorcycle - Green
                5: (255, 0, 255),    # Bus - Magenta
                7: (0, 165, 255)     # Truck - Orange
            }
            
            for box, track_id, cls, conf in zip(boxes, ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
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
        
        # Hiển thị thông tin device
        device_text = f'Device: {self.device.upper()}'
        if self.use_half:
            device_text += ' (FP16)'
        cv2.putText(frame, device_text, (20, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        """Xử lý một frame - luôn chạy inference trên mọi frame để đảm bảo độ chính xác"""
        original_height, original_width = frame.shape[:2]
        
        # Resize frame để giảm độ phân giải inference (tăng tốc đáng kể)
        # Nhưng vẫn giữ tỷ lệ khung hình
        scale_x = 1.0
        scale_y = 1.0
        
        if original_width > self.inference_size or original_height > self.inference_size:
            # Tính scale để fit vào inference_size nhưng giữ tỷ lệ
            scale = min(self.inference_size / original_width, 
                       self.inference_size / original_height)
            inference_width = int(original_width * scale)
            inference_height = int(original_height * scale)
            inference_frame = cv2.resize(frame, (inference_width, inference_height), 
                                        interpolation=cv2.INTER_LINEAR)
            # Tính scale factors chính xác
            scale_x = original_width / inference_width
            scale_y = original_height / inference_height
        else:
            inference_frame = frame
        
        # Chạy YOLOv11 đã được huấn luyện với tracking - luôn chạy trên mọi frame
        # QUAN TRỌNG: Không dùng imgsz parameter để mô hình tự xử lý kích thước
        # Boxes trả về sẽ theo kích thước inference_frame, sau đó chúng ta scale về gốc
        results = self.model.track(
            inference_frame, 
            persist=True, 
            tracker="bytetrack.yaml",
            classes=self.vehicle_classes, 
            conf=0.25,
            device=self.device,
            half=self.use_half,  # Sử dụng FP16 nếu có GPU
            verbose=False  # Tắt output để tăng tốc
        )
        
        # Cập nhật số lượng
        self.update_counts(results, original_height, scale_x, scale_y)
        
        # Vẽ kết quả
        annotated_frame = self.draw_results(frame, results, scale_x, scale_y)
        
        return annotated_frame
    
    def reset_counts(self):
        """Reset bộ đếm"""
        self.count_up = 0
        self.count_down = 0
        self.tracks.clear()
        self.last_update.clear()
        self.class_counts = {
            cls: {'up': 0, 'down': 0}
            for cls in self.vehicle_classes
        }

    def get_class_counts(self):
        """
        Trả về số lượng đếm theo từng loại xe.
        Returns:
            dict: {class_name: {'up': int, 'down': int, 'total': int}}
        """
        summary = {}
        for cls, counts in self.class_counts.items():
            name = self.class_names.get(cls, str(cls))
            up = counts['up']
            down = counts['down']
            summary[name] = {
                'up': up,
                'down': down,
                'total': up + down
            }
        return summary
