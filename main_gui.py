import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
import os
from vehicle_counter import VehicleCounter

class VehicleCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống đếm phương tiện giao thông")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2b2b2b')
        
        # Khởi tạo vehicle counter
        self.counter = None
        self.video_source = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.processed_video_path = None  # Đường dẫn video đã xử lý
        
        # Cấu hình hiệu năng
        self.inference_size = 640  # Kích thước inference (có thể điều chỉnh)
        self.target_fps = 30  # FPS mục tiêu
        self.display_fps = 15  # FPS hiển thị (giảm để tăng tốc)
        
        # Queue cho frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        
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
        
        # Nút xử lý video trước (preprocessing)
        self.btn_preprocess = tk.Button(control_frame, text="⚡ Xử lý video trước", 
                                        command=self.preprocess_video,
                                        bg='#9C27B0', fg='white',
                                        font=('Arial', 12), 
                                        relief=tk.RAISED, bd=3,
                                        width=20, height=2,
                                        state=tk.DISABLED)
        self.btn_preprocess.pack(pady=10)
        
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
        self.line_scale.set(0.7)
        self.line_scale.pack(pady=5)
        
        # Separator
        separator2 = tk.Frame(control_frame, height=2, bg='#555555')
        separator2.pack(fill=tk.X, padx=20, pady=10)
        
        # Cài đặt hiệu năng
        perf_frame = tk.Frame(control_frame, bg='#3c3c3c')
        perf_frame.pack(pady=10)
        
        tk.Label(perf_frame, text="⚡ Cài đặt hiệu năng", 
                bg='#3c3c3c', fg='#FFD700',
                font=('Arial', 10, 'bold')).pack()
        
        tk.Label(perf_frame, text="Độ phân giải inference:", 
                bg='#3c3c3c', fg='white', 
                font=('Arial', 9)).pack(pady=(10, 5))
        
        self.size_var = tk.StringVar(value="640")
        size_options = [("320 (Nhanh nhất)", "320"), 
                       ("640 (Cân bằng)", "640"),
                       ("960 (Chính xác hơn)", "960")]
        for text, value in size_options:
            tk.Radiobutton(perf_frame, text=text, variable=self.size_var, 
                          value=value, bg='#3c3c3c', fg='white',
                          selectcolor='#555555', activebackground='#3c3c3c',
                          activeforeground='white', font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        
        tk.Label(perf_frame, text="FPS hiển thị:", 
                bg='#3c3c3c', fg='white', 
                font=('Arial', 9)).pack(pady=(10, 5))
        
        self.fps_var = tk.StringVar(value="15")
        fps_options = [("10 FPS (Rất nhanh)", "10"), 
                      ("15 FPS (Nhanh)", "15"),
                      ("30 FPS (Mượt)", "30")]
        for text, value in fps_options:
            tk.Radiobutton(perf_frame, text=text, variable=self.fps_var, 
                          value=value, bg='#3c3c3c', fg='white',
                          selectcolor='#555555', activebackground='#3c3c3c',
                          activeforeground='white', font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        
        
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
            self.btn_preprocess.config(state=tk.NORMAL)
            self.processed_video_path = None  # Reset processed video
            
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
        
        # Lấy cài đặt hiệu năng
        self.inference_size = int(self.size_var.get())
        self.display_fps = int(self.fps_var.get())
        
        # Nếu có video đã xử lý, sử dụng nó
        if self.processed_video_path and os.path.exists(self.processed_video_path):
            self.video_source = self.processed_video_path
            self.status_label.config(text="Đang phát video đã xử lý...", fg='#4CAF50')
        
        # Khởi tạo vehicle counter nếu chưa có hoặc cần cập nhật cài đặt
        if self.counter is None:
            try:
                self.counter = VehicleCounter(
                    model_path='best.pt',
                    line_position=self.line_scale.get(),
                    inference_size=self.inference_size,
                    use_half_precision=True
                )
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải model!\n{str(e)}")
                return
        else:
            # Cập nhật cài đặt hiệu năng
            self.counter.inference_size = self.inference_size
        
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
        """Xử lý video trong thread riêng với tối ưu hóa hiệu năng"""
        import time
        
        # Nếu là video đã xử lý, chỉ cần phát lại nhanh
        is_processed = (self.processed_video_path and 
                       os.path.exists(self.processed_video_path) and 
                       self.video_source == self.processed_video_path)
        
        if is_processed:
            # Video đã xử lý - chỉ phát lại, không cần inference
            frame_time = 1.0 / self.display_fps
            last_time = time.time()
            frames_processed = 0
            
            while self.is_running and self.cap.isOpened():
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                
                ret, frame = self.cap.read()
                if not ret:
                    self.is_running = False
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Thông báo", "Video đã kết thúc!"))
                    break
                
                # Chỉ hiển thị, không xử lý
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = self.resize_frame(frame_rgb, 1000, 600)
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image=image)
                self.root.after(0, self.update_frame, photo)
                
                if frames_processed % 5 == 0:
                    self.root.after(0, self.update_stats)
                
                frames_processed += 1
                last_time = time.time()
        else:
            # Video chưa xử lý - xử lý real-time với tối ưu hóa
            frame_time = 1.0 / self.display_fps  # Sử dụng display_fps thay vì target_fps
            last_time = time.time()
            frames_processed = 0
            start_time = time.time()
            frame_skip_display = max(1, int(30 / self.display_fps))  # Skip frame hiển thị để đạt FPS
            
            while self.is_running and self.cap.isOpened():
                current_time = time.time()
                
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.video_source == 0:  # Webcam
                        continue
                    else:  # Video file đã hết
                        self.is_running = False
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Thông báo", "Video đã kết thúc!"))
                        break
                
                # QUAN TRỌNG: Luôn xử lý mọi frame để đếm chính xác
                if self.counter:
                    frame = self.counter.process_frame(frame)
                    self.counter.line_position = self.line_scale.get()
                
                # Chỉ hiển thị mỗi N frame để tăng tốc
                if frames_processed % frame_skip_display == 0:
                    # Điều chỉnh tốc độ để đạt FPS hiển thị
                    elapsed = current_time - last_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                    
                    # Chuyển đổi frame để hiển thị
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame để phù hợp với cửa sổ
                    display_frame = self.resize_frame(frame_rgb, 1000, 600)
                    
                    # Chuyển đổi sang ImageTk
                    image = Image.fromarray(display_frame)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    # Cập nhật UI trong main thread
                    self.root.after(0, self.update_frame, photo)
                    last_time = time.time()
                
                # Cập nhật stats mỗi 5 frame để giảm overhead
                if frames_processed % 5 == 0:
                    self.root.after(0, self.update_stats)
                
                frames_processed += 1
                
                # Hiển thị FPS thực tế mỗi giây
                if frames_processed % 30 == 0:
                    elapsed_total = time.time() - start_time
                    actual_fps = frames_processed / elapsed_total if elapsed_total > 0 else 0
                    self.root.after(0, lambda f=actual_fps: self.status_label.config(
                        text=f"Đang xử lý... ({f:.1f} FPS)", fg='#4CAF50'))
        
        # Đóng video khi kết thúc
        if self.cap:
            self.cap.release()
        self.root.after(0, self.stop_processing)
    
    def preprocess_video(self):
        """Xử lý video trước và lưu kết quả"""
        if self.video_source is None or isinstance(self.video_source, int):
            messagebox.showerror("Lỗi", "Vui lòng chọn video file trước!")
            return
        
        # Hỏi người dùng có muốn xử lý không
        if not messagebox.askyesno("Xác nhận", 
                                   "Xử lý video trước sẽ mất thời gian nhưng phát lại sẽ rất nhanh.\n"
                                   "Bạn có muốn tiếp tục?"):
            return
        
        # Lấy cài đặt hiệu năng
        self.inference_size = int(self.size_var.get())
        
        # Khởi tạo vehicle counter
        if self.counter is None:
            try:
                self.counter = VehicleCounter(
                    model_path='best.pt',
                    line_position=self.line_scale.get(),
                    inference_size=self.inference_size,
                    use_half_precision=True
                )
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải model!\n{str(e)}")
                return
        
        # Tạo đường dẫn file output
        base_name = os.path.splitext(self.video_source)[0]
        self.processed_video_path = f"{base_name}_processed.mp4"
        
        # Chạy preprocessing trong thread riêng
        self.btn_preprocess.config(state=tk.DISABLED)
        self.status_label.config(text="Đang xử lý video...", fg='#FF9800')
        
        thread = threading.Thread(target=self._preprocess_video_thread, daemon=True)
        thread.start()
    
    def _preprocess_video_thread(self):
        """Thread xử lý video trước"""
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                raise Exception("Không thể mở video")
            
            # Lấy thông tin video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Tạo video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.processed_video_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Xử lý frame
                processed_frame = self.counter.process_frame(frame)
                out.write(processed_frame)
                
                frame_count += 1
                
                # Cập nhật progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.root.after(0, lambda p=progress: self.status_label.config(
                        text=f"Đang xử lý... {p:.1f}%", fg='#FF9800'))
                    self.root.after(0, self.update_stats)
            
            cap.release()
            out.release()
            
            # Hoàn thành
            self.root.after(0, lambda: self.status_label.config(
                text=f"Đã xử lý xong! ({frame_count} frames)", fg='#4CAF50'))
            self.root.after(0, lambda: messagebox.showinfo(
                "Thành công", 
                f"Đã xử lý xong video!\n"
                f"File đã lưu: {self.processed_video_path}\n"
                f"Bấm 'Bắt đầu' để phát video đã xử lý."))
            self.root.after(0, lambda: self.btn_preprocess.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Lỗi", f"Lỗi khi xử lý video:\n{str(e)}"))
            self.root.after(0, lambda: self.status_label.config(
                text="Lỗi xử lý video", fg='#F44336'))
            self.root.after(0, lambda: self.btn_preprocess.config(state=tk.NORMAL))
        
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

