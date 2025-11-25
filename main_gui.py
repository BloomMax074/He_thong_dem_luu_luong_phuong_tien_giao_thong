import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from vehicle_counter import VehicleCounter

class VehicleCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống đếm phương tiện giao thông - YOLOv11")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2b2b2b')
        
        # Khởi tạo vehicle counter
        self.counter = None
        self.video_source = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        # Tạo giao diện
        self.create_widgets()
        
    def create_widgets(self):
        """Tạo các widget cho giao diện"""
        # Frame chính
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame điều khiển bên trái
        control_frame = tk.Frame(main_frame, bg='#3c3c3c', width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Tiêu đề
        title_label = tk.Label(control_frame, text="Điều khiển", 
                               font=('Arial', 16, 'bold'), 
                               bg='#3c3c3c', fg='white')
        title_label.pack(pady=20)
        
        # Nút chọn video
        btn_video = tk.Button(control_frame, text="📹 Chọn Video", 
                             command=self.select_video,
                             bg='#4CAF50', fg='white',
                             font=('Arial', 12), 
                             relief=tk.RAISED, bd=3,
                             width=20, height=2)
        btn_video.pack(pady=10)
        
        # Nút sử dụng Webcam
        btn_webcam = tk.Button(control_frame, text="📷 Sử dụng Webcam", 
                              command=self.use_webcam,
                              bg='#2196F3', fg='white',
                              font=('Arial', 12), 
                              relief=tk.RAISED, bd=3,
                              width=20, height=2)
        btn_webcam.pack(pady=10)
        
        # Nút bắt đầu
        self.btn_start = tk.Button(control_frame, text="▶ Bắt đầu", 
                                   command=self.start_processing,
                                   bg='#FF9800', fg='white',
                                   font=('Arial', 12, 'bold'), 
                                   relief=tk.RAISED, bd=3,
                                   width=20, height=2,
                                   state=tk.DISABLED)
        self.btn_start.pack(pady=10)
        
        # Nút dừng
        self.btn_stop = tk.Button(control_frame, text="⏸ Dừng", 
                                  command=self.stop_processing,
                                  bg='#F44336', fg='white',
                                  font=('Arial', 12), 
                                  relief=tk.RAISED, bd=3,
                                  width=20, height=2,
                                  state=tk.DISABLED)
        self.btn_stop.pack(pady=10)
        
        # Nút reset
        btn_reset = tk.Button(control_frame, text="🔄 Reset đếm", 
                             command=self.reset_count,
                             bg='#9E9E9E', fg='white',
                             font=('Arial', 12), 
                             relief=tk.RAISED, bd=3,
                             width=20, height=2)
        btn_reset.pack(pady=10)
        
        # Separator
        separator = tk.Frame(control_frame, height=2, bg='#555555')
        separator.pack(fill=tk.X, padx=20, pady=20)
        
        # Thông tin đường đếm
        line_frame = tk.Frame(control_frame, bg='#3c3c3c')
        line_frame.pack(pady=10)
        
        tk.Label(line_frame, text="Vị trí đường đếm:", 
                bg='#3c3c3c', fg='white', 
                font=('Arial', 10)).pack()
        
        self.line_scale = tk.Scale(line_frame, from_=0.1, to=0.9, 
                                   resolution=0.05, orient=tk.HORIZONTAL,
                                   bg='#3c3c3c', fg='white',
                                   troughcolor='#555555',
                                   command=self.update_line_position)
        self.line_scale.set(0.5)
        self.line_scale.pack(pady=5)
        
        # Hiển thị thông tin
        info_frame = tk.Frame(control_frame, bg='#3c3c3c')
        info_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.status_label = tk.Label(info_frame, text="Chưa khởi động", 
                                     bg='#3c3c3c', fg='#FFD700',
                                     font=('Arial', 10))
        self.status_label.pack()
        
        # Frame hiển thị video bên phải
        video_frame = tk.Frame(main_frame, bg='#1e1e1e')
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Label hiển thị video
        self.video_label = tk.Label(video_frame, text="Chưa có video",
                                    bg='#1e1e1e', fg='white',
                                    font=('Arial', 14))
        self.video_label.pack(expand=True)
        
        # Frame thống kê
        stats_frame = tk.Frame(video_frame, bg='#1e1e1e', height=80)
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        stats_frame.pack_propagate(False)
        
        # Labels thống kê
        self.stats_label = tk.Label(stats_frame, 
                                    text="Tổng số phương tiện: 0  |  Đi lên: 0  |  Đi xuống: 0",
                                    bg='#1e1e1e', fg='white',
                                    font=('Arial', 12, 'bold'))
        self.stats_label.pack(expand=True)
        
    def select_video(self):
        """Chọn file video"""
        file_path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.video_source = file_path
            self.status_label.config(text=f"Đã chọn: {file_path.split('/')[-1]}")
            self.btn_start.config(state=tk.NORMAL)
            
    def use_webcam(self):
        """Sử dụng webcam"""
        self.video_source = 0  # 0 = default webcam
        self.status_label.config(text="Đã chọn: Webcam")
        self.btn_start.config(state=tk.NORMAL)
        
    def update_line_position(self, value):
        """Cập nhật vị trí đường đếm"""
        if self.counter:
            self.counter.line_position = float(value)
            
    def start_processing(self):
        """Bắt đầu xử lý video/webcam"""
        if self.video_source is None:
            messagebox.showerror("Lỗi", "Vui lòng chọn video hoặc webcam trước!")
            return
        
        # Khởi tạo vehicle counter nếu chưa có
        if self.counter is None:
            try:
                self.counter = VehicleCounter(
                    model_path='yolo11n.pt',
                    line_position=self.line_scale.get()
                )
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải model YOLOv11!\n{str(e)}")
                return
        
        # Mở video/webcam
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise Exception("Không thể mở video/webcam")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở video source!\n{str(e)}")
            return
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="Đang xử lý...", fg='#4CAF50')
        
        # Bắt đầu thread xử lý
        self.process_thread = threading.Thread(target=self.process_video, daemon=True)
        self.process_thread.start()
        
    def stop_processing(self):
        """Dừng xử lý"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="Đã dừng", fg='#F44336')
        
    def reset_count(self):
        """Reset bộ đếm"""
        if self.counter:
            self.counter.reset_counts()
            self.update_stats()
            messagebox.showinfo("Thông báo", "Đã reset bộ đếm!")
        
    def process_video(self):
        """Xử lý video trong thread riêng"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                if self.video_source == 0:  # Webcam
                    continue
                else:  # Video file đã hết
                    self.is_running = False
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Thông báo", "Video đã kết thúc!"))
                    break
            
            # Xử lý frame với vehicle counter
            if self.counter:
                frame = self.counter.process_frame(frame)
            
            # Cập nhật vị trí đường đếm nếu thay đổi
            if self.counter:
                self.counter.line_position = self.line_scale.get()
            
            # Chuyển đổi frame để hiển thị
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame để phù hợp với cửa sổ
            display_frame = self.resize_frame(frame_rgb, 1000, 600)
            
            # Chuyển đổi sang ImageTk
            image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Cập nhật UI trong main thread
            self.root.after(0, self.update_frame, photo)
            self.root.after(0, self.update_stats)
            
            # Điều chỉnh FPS (30 FPS)
            cv2.waitKey(33)
        
        # Đóng video khi kết thúc
        if self.cap:
            self.cap.release()
        self.root.after(0, self.stop_processing)
        
    def resize_frame(self, frame, max_width, max_height):
        """Resize frame để phù hợp với kích thước hiển thị"""
        height, width = frame.shape[:2]
        
        # Tính tỷ lệ
        width_ratio = max_width / width
        height_ratio = max_height / height
        ratio = min(width_ratio, height_ratio, 1.0)
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return cv2.resize(frame, (new_width, new_height))
        
    def update_frame(self, photo):
        """Cập nhật frame hiển thị"""
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Giữ reference
        
    def update_stats(self):
        """Cập nhật thống kê"""
        if self.counter:
            total = self.counter.count_up + self.counter.count_down
            stats_text = (f"Tổng số phương tiện: {total}  |  "
                         f"Đi lên: {self.counter.count_up}  |  "
                         f"Đi xuống: {self.counter.count_down}")
            self.stats_label.config(text=stats_text)
        else:
            self.stats_label.config(text="Tổng số phương tiện: 0  |  Đi lên: 0  |  Đi xuống: 0")
        
    def on_closing(self):
        """Xử lý khi đóng cửa sổ"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCountingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

