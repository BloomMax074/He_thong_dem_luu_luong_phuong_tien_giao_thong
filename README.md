# Khung dàn ý báo cáo hệ thống đếm phương tiện giao thông (YOLOv11)

> Tài liệu này cung cấp dàn ý chi tiết để bạn viết báo cáo hoàn chỉnh. Mỗi mục nên bổ sung hình ảnh minh họa (ảnh giao diện, ảnh bounding boxes, đồ thị), số liệu (FPS, độ chính xác), và trích dẫn nguồn.

---

## 1. Giới thiệu
- Bối cảnh và nhu cầu: tắc nghẽn giao thông, nhu cầu giám sát tự động.
- Mục tiêu: đếm lưu lượng phương tiện theo hướng, hỗ trợ video và webcam, giao diện dễ dùng.
- Phạm vi: bài toán đếm 4 lớp phương tiện chính (car, motorcycle, bus, truck) dựa trên YOLOv11 + ByteTrack.
- Kết quả kỳ vọng: tốc độ xử lý đạt X FPS (GPU/CPU), độ chính xác đếm, khả năng phát lại nhanh với video đã xử lý.

## 2. Liên quan nghiên cứu
- Tổng quan các phương pháp đếm hiện có: nền tảng CNN/YOLO, kết hợp tracking (SORT, DeepSORT, ByteTrack).
- Lý do chọn YOLOv11: tốc độ, hỗ trợ tracker tích hợp, dễ triển khai.
- Lý do chọn ByteTrack: tracking ID ổn định, tránh đếm trùng.
- So sánh ngắn gọn với các lựa chọn khác (ví dụ: YOLOv8, YOLO-NAS) nếu cần.

## 3. Phân tích yêu cầu
- Chức năng: phát video/webcam, nhận diện + đếm theo hướng, hiển thị kết quả, cho phép xử lý trước để phát lại nhanh.
- Phi chức năng: hiệu năng (FPS tối thiểu mong muốn), độ trễ hiển thị, tính ổn định, dễ sử dụng (GUI).
- Ràng buộc: phần cứng (GPU/CPU), độ phân giải đầu vào, dung lượng RAM, thời gian xử lý cho video dài.

## 4. Kiến trúc hệ thống
- Sơ đồ khối:
  - Input (video/webcam) → YOLOv11 detect → ByteTrack track → Counting logic → Hiển thị/ghi video.
- Thành phần chính:
  - `vehicle_counter.py`: model YOLOv11, tracking ByteTrack, logic đếm (line crossing).
  - `main_gui.py`: giao diện Tkinter, điều khiển chọn nguồn, cấu hình, hiển thị, xử lý trước video.
  - `requirements.txt`: thư viện phụ thuộc.
- Luồng dữ liệu: từ frame đầu vào → inference (resize + FP16 nếu GPU) → tracking ID → cập nhật bộ đếm → render overlay → (tùy chọn) ghi ra file.

## 5. Thiết kế chi tiết
- Nhận diện: YOLOv11 với danh sách lớp COCO `[2,3,5,7]`.
- Tracking: ByteTrack với `persist=True` để giữ ID giữa các frame.
- Counting line: tham số `line_position` (0–1 theo chiều cao), logic xác định hướng (up/down) dựa trên trung điểm bbox qua hai frame liên tiếp.
- Hiển thị: vẽ đường đếm, bbox theo màu lớp, label gồm tên lớp, ID, confidence, hướng; hiển thị tổng/đi lên/đi xuống.
- Xử lý hiệu năng:
  - Resize frame về `inference_size` (320/640/960), scale ngược bbox về kích thước gốc.
  - FP16 khi có GPU.
  - Tùy chọn giảm FPS hiển thị (10/15/30) để UI mượt hơn.
- Xử lý trước (preprocessing):
  - Chạy toàn bộ video, ghi `*_processed.mp4` với overlay và kết quả đếm; phát lại nhanh vì không cần inference.

## 6. Triển khai
- Môi trường: Python 3.8+, torch/ultralytics/opencv/pillow.
- Hướng dẫn cài đặt: `pip install -r requirements.txt`; tải model tự động lần đầu.
- Cấu trúc thư mục:
  - `main_gui.py`: GUI, điều khiển luồng.
  - `vehicle_counter.py`: mô hình + logic đếm.
  - `best.pt` / `yolo11n.pt`: trọng số.
  - `README.md`: tài liệu.
- Giao diện người dùng:
  - Chọn video/webcam, chọn độ phân giải inference, chọn FPS hiển thị, nút “Xử lý video trước”, start/stop/reset.

## 7. Thử nghiệm và đánh giá
- Thiết lập thử nghiệm:
  - Phần cứng: GPU/CPU, RAM.
  - Bộ video thử: độ phân giải, độ dài, mật độ xe.
- Chỉ số đánh giá:
  - FPS xử lý real-time (GPU/CPU) ở các mức 320/640/960.
  - Độ chính xác đếm (so sánh đếm tự động vs đếm tay trên một đoạn video chuẩn).
  - Tốc độ phát lại video đã xử lý.
- Kết quả (điền số liệu đo được):
  - Bảng FPS theo cấu hình.
  - Bảng sai số đếm (% lệch).
  - Thời gian xử lý toàn bộ video khi preprocessing.
- Phân tích: nhận xét nguyên nhân chậm (CPU/GPU), ảnh hưởng của độ phân giải và FPS hiển thị.

## 8. Khó khăn và cách khắc phục
- Hiệu năng chậm trên CPU → giảm `inference_size`, giảm FPS hiển thị, ưu tiên preprocessing.
- Tracking mất ID khi vật thể nhanh/che khuất → tolerance line-crossing, giữ `persist=True`, giảm nhiễu bằng scale chính xác.
- Giới hạn bộ nhớ khi video dài → xử lý theo lô hoặc preprocessing lưu ra file.

## 9. Hướng phát triển
- Hỗ trợ nhiều đường đếm và nhiều khu vực.
- Xuất báo cáo CSV/Excel, biểu đồ theo thời gian.
- Cảnh báo bất thường (dừng/đi ngược chiều).
- Tối ưu mô hình nhẹ hơn (TensorRT, ONNX, YOLOv11n/int8).
- Triển khai edge device (Jetson, Raspberry Pi + NPU).

## 10. Kết luận
- Tóm tắt mục tiêu đã đạt: đếm theo hướng, GUI thân thiện, hai chế độ (real-time tối ưu, preprocessing phát lại nhanh).
- Đánh giá ngắn gọn hiệu năng và độ chính xác.
- Đề xuất hướng cải tiến tiếp theo.

## Phụ lục (nên bổ sung khi viết báo cáo)
- Hình ảnh giao diện, ví dụ kết quả nhận diện/đếm.
- Đoạn mã quan trọng: gọi YOLO/ByteTrack, logic line-crossing.
- Hướng dẫn chạy nhanh:
  - `python main_gui.py`
  - Chọn video/webcam, chọn độ phân giải, chọn FPS hiển thị.
  - (Tùy chọn) bấm “Xử lý video trước” để có file phát lại nhanh.

---

## Phụ lục A – Tóm tắt hệ thống & hướng dẫn sử dụng (giữ lại cho báo cáo)

### A1. Tính năng chính
- Nhận diện & đếm 4 lớp phương tiện: car, motorcycle, bus, truck.
- Đếm theo hướng (lên/xuống) với đường đếm tùy chỉnh.
- Tracking bằng ByteTrack để tránh đếm trùng.
- Hai chế độ:
  - Xử lý real-time (có thể giảm FPS hiển thị để mượt).
  - Xử lý trước toàn bộ video → phát lại rất nhanh (không cần inference).
- Giao diện Tkinter: chọn video/webcam, chọn độ phân giải (320/640/960), chọn FPS hiển thị, start/stop/reset, xử lý video trước.

### A2. Yêu cầu hệ thống
- Python 3.8+; các thư viện: torch/ultralytics/opencv/pillow (cài qua `pip install -r requirements.txt`).
- GPU khuyến nghị; vẫn chạy được CPU nhưng chậm hơn.
- Webcam (nếu dùng chế độ webcam).

### A3. Hướng dẫn sử dụng giao diện
1. Chọn nguồn:
   - “📹 Chọn Video” để chọn file, hoặc “📷 Sử dụng Webcam”.
2. Cấu hình:
   - Chọn độ phân giải inference (320/640/960).
   - Chọn FPS hiển thị (10/15/30) nếu muốn tối ưu hiển thị.
3. Chạy:
   - Nhấn “▶ Bắt đầu” để xử lý; hệ thống hiển thị bbox, ID, hướng, số đếm.
   - Thanh trượt “Vị trí đường đếm” để chỉnh line (0–1 theo chiều cao).
4. Xử lý trước (tùy chọn, để phát lại nhanh):
   - Nhấn “⚡ Xử lý video trước”, chờ hoàn tất, sau đó “Bắt đầu” để phát file `*_processed.mp4`.
5. Dừng/Reset:
   - “⏸ Dừng” để dừng, “🔄 Reset đếm” để về 0.

### A4. Ghi chú hiệu năng
- Độ phân giải 320 + FPS hiển thị 10–15 cho CPU; 640/960 cho GPU.
- Preprocessing phù hợp video dài: inference một lần, phát lại nhanh.
- FP16 tự kích hoạt khi có GPU.

### A5. Cấu trúc dự án
```
He_thong_dem_luu_luong_phuong_tien_giao_thong/
├── main_gui.py          # Giao diện Tkinter, điều khiển luồng, preprocessing
├── vehicle_counter.py   # YOLOv11 + ByteTrack + logic đếm (line crossing)
├── requirements.txt     # Thư viện phụ thuộc
├── best.pt / yolo11n.pt # Trọng số model
└── README.md            # Tài liệu & dàn ý báo cáo
```

### A6. Chi tiết kỹ thuật
- Mô hình: YOLOv11 (Ultralytics), trọng số nhẹ `yolo11n.pt` hoặc `best.pt`.
- Tracking: ByteTrack với `persist=True` để giữ ID ổn định giữa các frame.
- Lớp phương tiện: COCO IDs `[2,3,5,7]` (car, motorcycle, bus, truck).
- Đường đếm: `line_position` (0–1 theo chiều cao), so sánh trung điểm bbox giữa hai frame để xác định hướng.
- Scale kích thước: resize về `inference_size` (320/640/960), scale ngược bbox về kích thước gốc trước khi đếm.
- Hiển thị: vẽ line, bbox theo màu lớp, label (class, ID, conf, direction), thống kê tổng/đi lên/đi xuống.
- Hiệu năng: FP16 khi có GPU; giảm FPS hiển thị (10/15/30) để UI mượt; preprocessing để phát lại nhanh.

### A7. Bộ dữ liệu sử dụng
- Link: "https://universe.roboflow.com/fsmvu/street-view-gdogo".
