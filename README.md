# Hệ thống đếm phương tiện giao thông sử dụng YOLOv11

Hệ thống đếm lưu lượng phương tiện giao thông tự động sử dụng mô hình YOLOv11, hỗ trợ đếm từ video file và webcam với giao diện đồ họa thân thiện.

## Tính năng

- ✅ Nhận diện và đếm phương tiện giao thông (ô tô, xe máy, xe bus, xe tải)
- ✅ Hỗ trợ cả video file và webcam
- ✅ Giao diện đồ họa dễ sử dụng
- ✅ Đếm phương tiện theo hướng (đi lên/đi xuống)
- ✅ Điều chỉnh vị trí đường đếm
- ✅ Tracking phương tiện để tránh đếm trùng
- ✅ Hiển thị real-time với bounding boxes và thông tin

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam (nếu sử dụng chức năng webcam)
- GPU (khuyến nghị, nhưng có thể chạy trên CPU)

## Cài đặt

1. **Clone repository hoặc tải mã nguồn**

2. **Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```

3. **Tải mô hình YOLOv11:**
   - Mô hình sẽ tự động được tải xuống khi chạy lần đầu tiên
   - Hoặc bạn có thể tải thủ công từ [Ultralytics](https://github.com/ultralytics/ultralytics)

## Sử dụng

### Chạy ứng dụng GUI:

```bash
python main_gui.py
```

### Hướng dẫn sử dụng giao diện:

1. **Chọn nguồn video:**
   - Nhấn **"📹 Chọn Video"** để chọn file video từ máy tính
   - Hoặc nhấn **"📷 Sử dụng Webcam"** để sử dụng camera

2. **Bắt đầu đếm:**
   - Nhấn **"▶ Bắt đầu"** để bắt đầu xử lý
   - Hệ thống sẽ tự động nhận diện và đếm phương tiện

3. **Điều chỉnh đường đếm:**
   - Sử dụng thanh trượt **"Vị trí đường đếm"** để điều chỉnh vị trí đường đếm
   - Phương tiện sẽ được đếm khi vượt qua đường này

4. **Xem kết quả:**
   - Số lượng phương tiện được hiển thị trên màn hình và trong phần thống kê
   - Bạn có thể xem số lượng đi lên, đi xuống và tổng số

5. **Dừng/Reset:**
   - Nhấn **"⏸ Dừng"** để dừng xử lý
   - Nhấn **"🔄 Reset đếm"** để reset bộ đếm về 0

## Cấu trúc dự án

```
Vehicle_Detection&Counting/
│
├── main_gui.py          # Giao diện chính của ứng dụng
├── vehicle_counter.py   # Module xử lý đếm phương tiện
├── requirements.txt     # Danh sách thư viện cần thiết
└── README.md           # Hướng dẫn sử dụng
```

## Chi tiết kỹ thuật

### Mô hình sử dụng:
- **YOLOv11** (Ultralytics) - Mô hình nhận diện đối tượng
- **ByteTrack** - Tracker để theo dõi phương tiện
- **OpenCV** - Xử lý video và webcam

### Loại phương tiện được nhận diện:
- Car (Ô tô)
- Motorcycle (Xe máy)
- Bus (Xe bus)
- Truck (Xe tải)

### Cách hoạt động:
1. Mô hình YOLOv11 nhận diện các phương tiện trong từng frame
2. ByteTrack theo dõi các phương tiện qua các frame
3. Hệ thống xác định khi phương tiện vượt qua đường đếm
4. Đếm phương tiện dựa trên hướng di chuyển

## Tùy chỉnh

### Điều chỉnh độ tin cậy (confidence threshold):
Trong file `vehicle_counter.py`, dòng 80:
```python
results = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                           classes=self.vehicle_classes, conf=0.25)
```
Thay đổi `conf=0.25` thành giá trị mong muốn (0.0 - 1.0)

### Thay đổi loại phương tiện:
Trong file `vehicle_counter.py`, dòng 25:
```python
self.vehicle_classes = [2, 3, 5, 7]  # COCO classes
```
Tham khảo COCO class list để thêm/bớt loại phương tiện.

## Xử lý lỗi thường gặp

1. **Lỗi không tải được model:**
   - Kiểm tra kết nối internet (để tải model lần đầu)
   - Hoặc tải model thủ công và đặt vào thư mục dự án

2. **Lỗi không mở được webcam:**
   - Kiểm tra webcam đã được kết nối
   - Thử thay đổi `self.video_source = 0` thành `1`, `2`, ... trong code

3. **Hiệu năng chậm:**
   - Giảm độ phân giải video
   - Sử dụng GPU nếu có
   - Giảm confidence threshold

## Phát triển thêm

Một số tính năng có thể thêm:
- Lưu kết quả vào file CSV/Excel
- Export video với kết quả đếm
- Vẽ biểu đồ thống kê theo thời gian
- Hỗ trợ nhiều đường đếm
- Gửi cảnh báo khi có sự cố

## License

Dự án này được phát hành dưới giấy phép MIT.

## Tác giả

Hệ thống được xây dựng với YOLOv11 từ Ultralytics.

## Tham khảo

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)

