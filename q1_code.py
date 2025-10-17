import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import threading
from datetime import datetime
import math
import glob

class FuturisticLicensePlateDetector:
    def __init__(self):
        # Initialize model
        self.model = None
        self.model_status = "⚡ INITIALIZING NEURAL NETWORK..."
        self.load_model()
        
        # License plate damage classes
        self.classes = ['broken', 'non broken']
        
        # Cyberpunk color scheme
        self.colors = {
            'primary': '#0a0a12',
            'secondary': '#1a1a2e',
            'accent': '#00ffff',
            'accent2': '#ff00ff',
            'success': '#00ff88',
            'warning': '#ffaa00',
            'background': '#050510',
            'card_bg': '#0f0f1a',
            'text_primary': '#e0e0ff',
            'text_secondary': '#a0a0cc',
            'neon_blue': '#00ffff',
            'neon_pink': '#ff00ff',
            'neon_green': '#00ff88',
            'grid': '#1a1a3a'
        }
        
        # Application state
        self.current_files = []
        self.current_index = 0
        self.processing = False
        self.animation_phase = 0
        self.batch_results = []
        
        self.setup_gui()
        self.start_animations()

    def load_model(self):
        """Load YOLO model in a separate thread"""
        def load_model_thread():
            try:
                self.model = YOLO("runs/detect/train/weights/best.pt")
                self.model_status = "✅ NEURAL NETWORK ONLINE"
            except Exception as e:
                self.model_status = f"❌ SYSTEM ERROR: {str(e)}"
                try:
                    self.model = YOLO("yolov8n.pt")
                    self.model_status = "⚠️ RUNNING ON DEFAULT CORE"
                except:
                    self.model_status = "❌ CRITICAL SYSTEM FAILURE"
        
        thread = threading.Thread(target=load_model_thread)
        thread.daemon = True
        thread.start()

    def setup_gui(self):
        """Setup futuristic cyberpunk GUI"""
        self.root = tk.Tk()
        self.root.title("NEURAL PLATE SCANNER v2.0 - BATCH MODE")
        self.root.geometry("1400x900")
        self.root.configure(bg=self.colors['primary'], cursor='dotbox')
        self.root.minsize(1200, 800)
        
        # Make window slightly transparent for glass effect
        self.root.attributes('-alpha', 0.98)
        
        # Configure styles
        self.setup_styles()
        
        # Create main container with grid background
        main_container = ttk.Frame(self.root, style='Primary.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Header with animated elements
        header_frame = ttk.Frame(main_container, style='Secondary.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 15), padx=1)
        
        # Animated header content
        header_content = ttk.Frame(header_frame, style='Secondary.TFrame')
        header_content.pack(fill=tk.X, padx=30, pady=12)
        
        # Title with cyberpunk font simulation
        title_frame = ttk.Frame(header_content, style='Secondary.TFrame')
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(title_frame, 
                               text="▮ NEURAL PLATE SCANNER - BATCH MODE ▮", 
                               style='Title.TLabel')
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(title_frame, 
                                  text="> ADVANCED BATCH LICENSE PLATE DAMAGE DETECTION SYSTEM <", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(anchor=tk.W, pady=(2, 0))
        
        # System status indicator
        status_indicator = ttk.Frame(header_content, style='Secondary.TFrame')
        status_indicator.pack(anchor=tk.E, pady=(5, 0))
        
        self.system_status = ttk.Label(status_indicator, 
                                     text="■ SYSTEM: ONLINE", 
                                     style='Status.TLabel')
        self.system_status.pack(anchor=tk.E)
        
        # Main content area
        content_frame = ttk.Frame(main_container, style='Primary.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left control panel - Cyberdeck style
        cyberdeck_frame = ttk.Frame(content_frame, style='Cyberdeck.TFrame', width=350)
        cyberdeck_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        cyberdeck_frame.pack_propagate(False)
        
        # Right display area - Main terminal
        terminal_frame = ttk.Frame(content_frame, style='Terminal.TFrame')
        terminal_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ===== CYBERDECK CONTROLS =====
        
        # File Operations Terminal
        file_terminal = ttk.LabelFrame(cyberdeck_frame, text="[ BATCH OPERATIONS ]", style='CyberTerminal.TFrame')
        file_terminal.pack(fill=tk.X, padx=12, pady=(15, 8))
        
        # Upload buttons with cyberpunk style
        self.btn_single = ttk.Button(file_terminal, 
                                   text="▷ SCAN SINGLE IMAGE", 
                                   command=self.upload_single_image,
                                   style='CyberButton.TButton')
        self.btn_single.pack(fill=tk.X, padx=8, pady=6)
        
        self.btn_folder = ttk.Button(file_terminal, 
                                   text="▷ SCAN ENTIRE FOLDER", 
                                   command=self.upload_folder,
                                   style='CyberButton.TButton')
        self.btn_folder.pack(fill=tk.X, padx=8, pady=6)
        
        self.btn_video = ttk.Button(file_terminal, 
                                   text="▷ SCAN VIDEO", 
                                   command=self.upload_video,
                                   style='CyberButton.TButton')
        self.btn_video.pack(fill=tk.X, padx=8, pady=(0, 10))
        
        # Batch Navigation
        nav_frame = ttk.LabelFrame(cyberdeck_frame, text="[ BATCH NAVIGATION ]", style='CyberTerminal.TFrame')
        nav_frame.pack(fill=tk.X, padx=12, pady=8)
        
        nav_buttons_frame = ttk.Frame(nav_frame, style='CyberTerminal.TFrame')
        nav_buttons_frame.pack(fill=tk.X, padx=8, pady=8)
        
        self.btn_prev = ttk.Button(nav_buttons_frame, 
                                  text="◀ PREV", 
                                  command=self.previous_image,
                                  style='NavButton.TButton')
        self.btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        self.btn_next = ttk.Button(nav_buttons_frame, 
                                  text="NEXT ▶", 
                                  command=self.next_image,
                                  style='NavButton.TButton')
        self.btn_next.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        
        # Batch info
        self.batch_info = ttk.Label(nav_frame, text="NO BATCH LOADED", style='BatchInfo.TLabel')
        self.batch_info.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Analysis Core Settings
        core_frame = ttk.LabelFrame(cyberdeck_frame, text="[ ANALYSIS CORE ]", style='CyberTerminal.TFrame')
        core_frame.pack(fill=tk.X, padx=12, pady=8)
        
        # Neural network confidence
        conf_frame = ttk.Frame(core_frame, style='CyberTerminal.TFrame')
        conf_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(conf_frame, text="NEURAL CONFIDENCE:", style='CyberText.TLabel').pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.25)
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, 
                              variable=self.conf_var, orient=tk.HORIZONTAL,
                              style='Cyber.Horizontal.TScale')
        conf_scale.pack(fill=tk.X, pady=(8, 0))
        
        # Batch Statistics
        stats_frame = ttk.LabelFrame(cyberdeck_frame, text="[ BATCH STATISTICS ]", style='CyberTerminal.TFrame')
        stats_frame.pack(fill=tk.X, padx=12, pady=8)
        
        self.stats_text = tk.Text(stats_frame, height=4, width=25, 
                                 bg=self.colors['card_bg'], fg=self.colors['neon_blue'],
                                 font=('Courier New', 8), wrap=tk.WORD,
                                 relief='flat', padx=10, pady=10, borderwidth=0)
        self.stats_text.pack(fill=tk.BOTH, padx=5, pady=5)
        self.update_stats("AWAITING BATCH DATA...")
        
        # System Diagnostics Terminal
        diag_frame = ttk.LabelFrame(cyberdeck_frame, text="[ SYSTEM DIAGNOSTICS ]", style='CyberTerminal.TFrame')
        diag_frame.pack(fill=tk.X, padx=12, pady=8)
        
        self.status_text = tk.Text(diag_frame, height=6, width=25, 
                                  bg=self.colors['card_bg'], fg=self.colors['neon_green'],
                                  font=('Courier New', 8), wrap=tk.WORD,
                                  relief='flat', padx=10, pady=10, borderwidth=0,
                                  insertbackground=self.colors['neon_green'])
        self.status_text.pack(fill=tk.BOTH, padx=5, pady=5)
        self.update_status("> SYSTEM BOOT SEQUENCE INITIATED")
        
        # Detection Matrix Legend
        matrix_frame = ttk.LabelFrame(cyberdeck_frame, text="[ DETECTION MATRIX ]", style='CyberTerminal.TFrame')
        matrix_frame.pack(fill=tk.X, padx=12, pady=8)
        
        matrix_items = [
            ("■ CRITICAL DAMAGE", "BROKEN PLATE DETECTED", self.colors['neon_pink']),
            ("■ NO DAMAGE", "PLATE INTEGRITY CONFIRMED", self.colors['neon_green']),
            ("■ UNKNOWN", "ANOMALY DETECTED", self.colors['neon_blue'])
        ]
        
        for symbol, text, color in matrix_items:
            item_frame = ttk.Frame(matrix_frame, style='CyberTerminal.TFrame')
            item_frame.pack(fill=tk.X, padx=8, pady=4)
            
            # Pulsing indicator
            indicator = tk.Frame(item_frame, background=color, width=3, height=16)
            indicator.pack(side=tk.LEFT, padx=(0, 8))
            indicator.pack_propagate(False)
            
            # Matrix text
            text_frame = ttk.Frame(item_frame, style='CyberTerminal.TFrame')
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(text_frame, text=symbol, style='MatrixText.TLabel').pack(anchor=tk.W)
            ttk.Label(text_frame, text=text, style='CyberSmall.TLabel').pack(anchor=tk.W)
        
        # ===== MAIN TERMINAL DISPLAY =====
        
        # Results visualization terminal
        display_terminal = ttk.LabelFrame(terminal_frame, text="[ VISUALIZATION TERMINAL ]", style='Terminal.TFrame')
        display_terminal.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Main display with cyberpunk border
        self.panel = ttk.Label(display_terminal, 
                              text="NEURAL PLATE SCANNER READY\n\n"
                                   "> AWAITING INPUT DATA STREAM\n"
                                   "> SELECT BATCH MODULE TO BEGIN\n\n"
                                   "▮ SYSTEM STATUS: STANDBY",
                              style='TerminalText.TLabel',
                              anchor=tk.CENTER,
                              justify=tk.CENTER)
        self.panel.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Progress and system metrics
        metrics_frame = ttk.Frame(terminal_frame, style='Terminal.TFrame')
        metrics_frame.pack(fill=tk.X, pady=(8, 0))
        
        # Animated progress bar
        self.progress = ttk.Progressbar(metrics_frame, mode='determinate', style='Cyber.Horizontal.TProgressbar')
        self.progress.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 10))
        
        # System metrics
        self.metrics_label = ttk.Label(metrics_frame, text="READY", style='Metrics.TLabel')
        self.metrics_label.pack(side=tk.RIGHT)
        
        # Footer with system info
        footer_frame = ttk.Frame(main_container, style='Secondary.TFrame')
        footer_frame.pack(fill=tk.X, pady=(15, 0))
        
        footer_text = "◼ NEURAL NETWORK v2.0 ◼ BATCH PROCESSING MODE ◼ CYBERSCAN SYSTEMS ◼ PROPRIETARY TECHNOLOGY ◼"
        footer_label = ttk.Label(footer_frame, text=footer_text, style='Footer.TLabel')
        footer_label.pack(pady=6)

    def setup_styles(self):
        """Configure cyberpunk styles"""
        style = ttk.Style()
        style.theme_use('alt')
        
        # Configure frame styles
        style.configure('Primary.TFrame', 
                       background=self.colors['primary'],
                       relief='raised',
                       borderwidth=1)
        
        style.configure('Secondary.TFrame', 
                       background=self.colors['secondary'],
                       relief='sunken',
                       borderwidth=1)
        
        style.configure('Cyberdeck.TFrame',
                       background=self.colors['card_bg'],
                       relief='ridge',
                       borderwidth=2)
        
        style.configure('Terminal.TFrame',
                       background=self.colors['card_bg'],
                       relief='groove',
                       borderwidth=1)
        
        style.configure('CyberTerminal.TFrame',
                       background=self.colors['card_bg'],
                       relief='raised',
                       borderwidth=1)
        
        # Label styles
        style.configure('Title.TLabel',
                       background=self.colors['secondary'],
                       foreground=self.colors['neon_blue'],
                       font=('Courier New', 16, 'bold'),
                       borderwidth=0)
        
        style.configure('Subtitle.TLabel',
                       background=self.colors['secondary'],
                       foreground=self.colors['neon_pink'],
                       font=('Courier New', 9),
                       borderwidth=0)
        
        style.configure('Status.TLabel',
                       background=self.colors['secondary'],
                       foreground=self.colors['neon_green'],
                       font=('Courier New', 8, 'bold'),
                       borderwidth=0)
        
        style.configure('CyberText.TLabel',
                       background=self.colors['card_bg'],
                       foreground=self.colors['text_primary'],
                       font=('Courier New', 8),
                       borderwidth=0)
        
        style.configure('MatrixText.TLabel',
                       background=self.colors['card_bg'],
                       foreground=self.colors['neon_blue'],
                       font=('Courier New', 8, 'bold'),
                       borderwidth=0)
        
        style.configure('CyberSmall.TLabel',
                       background=self.colors['card_bg'],
                       foreground=self.colors['text_secondary'],
                       font=('Courier New', 7),
                       borderwidth=0)
        
        style.configure('Metrics.TLabel',
                       background=self.colors['card_bg'],
                       foreground=self.colors['neon_green'],
                       font=('Courier New', 8, 'bold'),
                       borderwidth=0)
        
        style.configure('BatchInfo.TLabel',
                       background=self.colors['card_bg'],
                       foreground=self.colors['neon_pink'],
                       font=('Courier New', 8, 'bold'),
                       borderwidth=0)
        
        style.configure('Footer.TLabel',
                       background=self.colors['secondary'],
                       foreground=self.colors['text_secondary'],
                       font=('Courier New', 7),
                       borderwidth=0)
        
        style.configure('TerminalText.TLabel',
                       background=self.colors['card_bg'],
                       foreground=self.colors['neon_green'],
                       font=('Courier New', 10),
                       relief='sunken',
                       borderwidth=1)
        
        # Button styles
        style.configure('CyberButton.TButton',
                       background=self.colors['secondary'],
                       foreground=self.colors['neon_blue'],
                       font=('Courier New', 9, 'bold'),
                       borderwidth=1,
                       relief='raised',
                       focuscolor='none')
        
        style.map('CyberButton.TButton',
                 background=[('active', self.colors['primary']),
                           ('pressed', self.colors['accent'])],
                 foreground=[('active', self.colors['neon_pink']),
                           ('pressed', self.colors['primary'])])
        
        style.configure('NavButton.TButton',
                       background=self.colors['card_bg'],
                       foreground=self.colors['neon_green'],
                       font=('Courier New', 8, 'bold'),
                       borderwidth=1,
                       relief='raised')
        
        style.map('NavButton.TButton',
                 background=[('active', self.colors['secondary']),
                           ('pressed', self.colors['neon_green'])],
                 foreground=[('active', self.colors['neon_pink']),
                           ('pressed', self.colors['primary'])])
        
        # Progressbar style
        style.configure('Cyber.Horizontal.TProgressbar',
                       background=self.colors['neon_blue'],
                       troughcolor=self.colors['secondary'],
                       borderwidth=0,
                       lightcolor=self.colors['neon_blue'],
                       darkcolor=self.colors['neon_blue'])
        
        # Scale style
        style.configure('Cyber.Horizontal.TScale',
                       background=self.colors['card_bg'],
                       troughcolor=self.colors['secondary'],
                       borderwidth=0)

    def start_animations(self):
        """Start cyberpunk animations"""
        def animate():
            self.animation_phase = (self.animation_phase + 0.1) % (2 * math.pi)
            glow = abs(math.sin(self.animation_phase)) * 0.3 + 0.7
            
            # Update system status with pulsing effect
            status_text = "■ SYSTEM: ONLINE" if self.animation_phase < math.pi else "▮ SYSTEM: ONLINE"
            self.system_status.config(text=status_text)
            
            # Schedule next animation frame
            self.root.after(100, animate)
        
        animate()

    def update_status(self, message):
        """Update status text with cyberpunk style"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"> [{timestamp}] {message}"
        
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"{formatted_message}\n\n{self.model_status}")
        self.status_text.config(state=tk.DISABLED)
        self.status_text.see(tk.END)
        self.root.update()

    def update_stats(self, message):
        """Update batch statistics"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, message)
        self.stats_text.config(state=tk.DISABLED)
        self.stats_text.see(tk.END)

    def process_batch(self, folder_path):
        """Process all images in a folder"""
        if self.model is None:
            messagebox.showerror("SYSTEM ERROR", "NEURAL NETWORK OFFLINE\nINITIATE REBOOT SEQUENCE")
            return

        try:
            # Get all image files from folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            self.current_files = []
            for extension in image_extensions:
                self.current_files.extend(glob.glob(os.path.join(folder_path, extension)))
                self.current_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
            
            if not self.current_files:
                raise Exception("NO VISUAL DATA FOUND IN SELECTED DIRECTORY")
            
            self.current_index = 0
            self.batch_results = []
            
            self.update_status(f"BATCH ACQUIRED: {len(self.current_files)} FILES")
            self.update_stats(f"BATCH SIZE: {len(self.current_files)}\nPROCESSING...")
            
            # Process first image
            self.process_current_image()
            
        except Exception as e:
            self.update_status(f"BATCH ERROR: {str(e)}")
            messagebox.showerror("BATCH ERROR", f"FAILED TO PROCESS BATCH:\n{str(e)}")

    def process_current_image(self):
        """Process and display current image in batch"""
        if not self.current_files or self.current_index >= len(self.current_files):
            return
        
        current_file = self.current_files[self.current_index]
        
        def process_image():
            image = self.predict_image(current_file)
            if image is None:
                return

            # Convert to PIL and resize
            image_pil = Image.fromarray(image)
            image_pil.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for display
            image_tk = ImageTk.PhotoImage(image_pil)
            self.panel.configure(image=image_tk, text="")
            self.panel.image = image_tk
            
            # Update batch info
            self.update_batch_info()
            
            # Update progress
            progress = (self.current_index + 1) / len(self.current_files) * 100
            self.progress.configure(value=progress)

        thread = threading.Thread(target=process_image)
        thread.daemon = True
        thread.start()

    def update_batch_info(self):
        """Update batch navigation information"""
        if self.current_files:
            current_file = os.path.basename(self.current_files[self.current_index])
            info_text = f"IMAGE {self.current_index + 1}/{len(self.current_files)}\n{current_file}"
            self.batch_info.config(text=info_text)
            
            # Update stats
            stats_text = f"BATCH: {len(self.current_files)} FILES\n"
            stats_text += f"CURRENT: {self.current_index + 1}/{len(self.current_files)}\n"
            stats_text += f"PROGRESS: {((self.current_index + 1) / len(self.current_files) * 100):.1f}%"
            self.update_stats(stats_text)

    def next_image(self):
        """Navigate to next image in batch"""
        if self.current_files and self.current_index < len(self.current_files) - 1:
            self.current_index += 1
            self.process_current_image()

    def previous_image(self):
        """Navigate to previous image in batch"""
        if self.current_files and self.current_index > 0:
            self.current_index -= 1
            self.process_current_image()

    def predict_image(self, image_path):
        """Perform image inference with cyberpunk visualization"""
        try:
            self.update_status(f"PROCESSING: {os.path.basename(image_path)}")
            self.metrics_label.config(text="ANALYZING...")
            
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("DATA STREAM CORRUPTED")

            # Perform inference
            results = self.model.predict(
                source=image_path, 
                conf=self.conf_var.get(),
                verbose=False
            )
            output = results[0]

            detection_count = 0
            for box, conf, cls in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)

                # Get class name with safety check
                class_name = self.classes[class_id] if class_id < len(self.classes) else f"ANOMALY_{class_id}"

                # Cyberpunk color coding
                if class_name == "broken":
                    color = (255, 0, 255)  # Neon Pink - Critical
                    label_color = (255, 255, 255)
                else:
                    color = (0, 255, 136)  # Neon Green - No damage
                    label_color = (0, 0, 0)

                # Draw bounding box with cyberpunk style
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Add corner markers for cyberpunk look
                marker_size = 8
                # Top-left
                cv2.line(image, (x1, y1), (x1 + marker_size, y1), color, 2)
                cv2.line(image, (x1, y1), (x1, y1 + marker_size), color, 2)
                # Top-right
                cv2.line(image, (x2, y1), (x2 - marker_size, y1), color, 2)
                cv2.line(image, (x2, y1), (x2, y1 + marker_size), color, 2)
                # Bottom-left
                cv2.line(image, (x1, y2), (x1 + marker_size, y2), color, 2)
                cv2.line(image, (x1, y2), (x1, y2 - marker_size), color, 2)
                # Bottom-right
                cv2.line(image, (x2, y2), (x2 - marker_size, y2), color, 2)
                cv2.line(image, (x2, y2), (x2, y2 - marker_size), color, 2)
                
                # Draw label with cyberpunk style
                label = f"{class_name.upper()}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Label background
                cv2.rectangle(image, (x1, y1 - label_size[1] - 15), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                
                # Label text
                cv2.putText(image, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                
                detection_count += 1

            self.update_status(f"ANALYSIS COMPLETE - {detection_count} TARGET(S)")
            self.metrics_label.config(text=f"TARGETS: {detection_count}")
            
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            self.update_status(f"PROCESSING ERROR: {str(e)}")
            self.metrics_label.config(text="PROCESS FAILED")
            return None

    def upload_single_image(self):
        """Handle single image upload"""
        file_path = filedialog.askopenfilename(
            title="SELECT VISUAL DATA SOURCE",
            filetypes=[
                ("VISUAL DATA FILES", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("ALL COMPATIBLE FORMATS", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"),
                ("ALL DATA STREAMS", "*.*")
            ]
        )
        if file_path:
            self.current_files = [file_path]
            self.current_index = 0
            filename = os.path.basename(file_path)
            self.update_status(f"SINGLE FILE ACQUIRED: {filename}")
            self.update_stats("MODE: SINGLE FILE\nPROCESSING...")
            self.process_current_image()

    def upload_folder(self):
        """Handle folder upload for batch processing"""
        folder_path = filedialog.askdirectory(title="SELECT BATCH DATA DIRECTORY")
        if folder_path:
            self.update_status(f"SCANNING DIRECTORY: {folder_path}")
            self.process_batch(folder_path)

    def upload_video(self):
        """Handle video upload"""
        file_path = filedialog.askopenfilename(
            title="SELECT TEMPORAL DATA STREAM",
            filetypes=[
                ("TEMPORAL DATA FILES", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("ALL COMPATIBLE STREAMS", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
                ("ALL DATA STREAMS", "*.*")
            ]
        )
        if file_path:
            filename = os.path.basename(file_path)
            self.update_status(f"TEMPORAL STREAM ACQUIRED: {filename}")
            messagebox.showinfo("SYSTEM NOTICE", "TEMPORAL ANALYSIS MODULE\nUNDER DEVELOPMENT")

    def run(self):
        """Start the application"""
        # Center the window on screen
        self.root.eval('tk::PlaceWindow . center')
        self.root.mainloop()

if __name__ == "__main__":
    app = FuturisticLicensePlateDetector()
    app.run()