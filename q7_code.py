import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import json
import os
import threading
from datetime import datetime

class ProfessionalCatDogClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Classifier Pro - Cat vs Dog Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f5f5f5')
        
        # Set window icon (you can add an icon file)
        try:
            self.root.iconbitmap("animal_icon.ico")  # Optional: add your icon file
        except:
            pass
        
        # Initialize model and variables
        self.model = None
        self.class_labels = {}
        self.image_paths = []
        self.current_image = None
        self.is_analyzing = False
        
        # Modern color scheme
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'accent': '#e74c3c',
            'success': '#27ae60',
            'warning': '#f39c12',
            'light': '#ecf0f1',
            'dark': '#34495e',
            'background': '#f5f5f5'
        }
        
        self.setup_styles()
        self.setup_model()
        self.create_gui()
    
    def setup_styles(self):
        """Configure modern styles for widgets"""
        style = ttk.Style()
        
        # Configure theme
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Primary.TFrame', background=self.colors['primary'])
        style.configure('Light.TFrame', background=self.colors['light'])
        style.configure('Background.TFrame', background=self.colors['background'])
        
        # Button styles
        style.configure('Primary.TButton', 
                       background=self.colors['secondary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        style.map('Primary.TButton',
                 background=[('active', self.colors['primary']),
                           ('pressed', self.colors['dark'])])
        
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground='white',
                       borderwidth=0)
        style.map('Accent.TButton',
                 background=[('active', '#c0392b'),
                           ('pressed', '#a93226')])
        
        # Label styles
        style.configure('Title.TLabel',
                       background=self.colors['background'],
                       foreground=self.colors['primary'],
                       font=('Segoe UI', 18, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=self.colors['background'],
                       foreground=self.colors['dark'],
                       font=('Segoe UI', 12))
        
        # Treeview styles
        style.configure('Custom.Treeview',
                       background='white',
                       fieldbackground='white',
                       foreground=self.colors['dark'],
                       rowheight=25)
        
        style.configure('Custom.Treeview.Heading',
                       background=self.colors['primary'],
                       foreground='white',
                       relief='flat',
                       font=('Segoe UI', 10, 'bold'))
    
    def setup_model(self):
        """Initialize the pre-trained model"""
        try:
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            self.load_imagenet_labels()
            
            # Preprocessing transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                               f"Failed to load AI model:\n{str(e)}")
    
    def load_imagenet_labels(self):
        """Load ImageNet class labels for cats and dogs"""
        # Cat classes
        cat_classes = {
            281: 'Tabby Cat', 282: 'Tiger Cat', 283: 'Persian Cat', 
            284: 'Siamese Cat', 285: 'Egyptian Cat'
        }
        
        # Dog classes (extensive list)
        dog_classes = {
            151: 'Chihuahua', 152: 'Japanese Spaniel', 153: 'Maltese', 
            154: 'Pekinese', 155: 'Shih-Tzu', 156: 'Blenheim Spaniel',
            157: 'Papillon', 158: 'Toy Terrier', 159: 'Rhodesian Ridgeback',
            160: 'Afghan Hound', 161: 'Basset Hound', 162: 'Beagle',
            163: 'Bloodhound', 164: 'Bluetick Coonhound', 165: 'Black-and-Tan Coonhound',
            166: 'Walker Hound', 167: 'English Foxhound', 168: 'Redbone Coonhound',
            169: 'Borzoi', 170: 'Irish Wolfhound', 171: 'Italian Greyhound',
            172: 'Whippet', 173: 'Ibizan Hound', 174: 'Norwegian Elkhound',
            175: 'Otterhound', 176: 'Saluki', 177: 'Scottish Deerhound',
            178: 'Weimaraner', 179: 'Staffordshire Bullterrier',
            180: 'American Staffordshire Terrier', 181: 'Bedlington Terrier',
            182: 'Border Terrier', 183: 'Kerry Blue Terrier', 184: 'Irish Terrier',
            185: 'Norfolk Terrier', 186: 'Norwich Terrier', 187: 'Yorkshire Terrier',
            188: 'Wire-haired Fox Terrier', 189: 'Lakeland Terrier',
            190: 'Sealyham Terrier', 191: 'Airedale Terrier', 192: 'Cairn Terrier',
            193: 'Australian Terrier', 194: 'Dandie Dinmont', 195: 'Boston Bull',
            196: 'Miniature Schnauzer', 197: 'Giant Schnauzer', 198: 'Standard Schnauzer',
            199: 'Scottish Terrier', 200: 'Tibetan Terrier', 201: 'Silky Terrier',
            202: 'Soft-coated Wheaten Terrier', 203: 'West Highland White Terrier',
            204: 'Lhasa', 205: 'Flat-coated Retriever', 206: 'Curly-coated Retriever',
            207: 'Golden Retriever', 208: 'Labrador Retriever', 209: 'Chesapeake Bay Retriever',
            210: 'German Short-haired Pointer', 211: 'Vizsla', 212: 'English Setter',
            213: 'Irish Setter', 214: 'Gordon Setter', 215: 'Brittany Spaniel',
            216: 'Clumber Spaniel', 217: 'English Springer Spaniel', 218: 'Welsh Springer Spaniel',
            219: 'Cocker Spaniel', 220: 'Sussex Spaniel', 221: 'Irish Water Spaniel',
            222: 'Kuvasz', 223: 'Schipperke', 224: 'Groenendael', 225: 'Malinois',
            226: 'Briard', 227: 'Kelpie', 228: 'Komondor', 229: 'Old English Sheepdog',
            230: 'Shetland Sheepdog', 231: 'Collie', 232: 'Border Collie',
            233: 'Bouvier des Flandres', 234: 'Rottweiler', 235: 'German Shepherd',
            236: 'Doberman', 237: 'Miniature Pinscher', 238: 'Greater Swiss Mountain Dog',
            239: 'Bernese Mountain Dog', 240: 'Appenzeller', 241: 'Entlebucher',
            242: 'Boxer', 243: 'Bull Mastiff', 244: 'Tibetan Mastiff', 245: 'French Bulldog',
            246: 'Great Dane', 247: 'Saint Bernard', 248: 'Eskimo Dog', 249: 'Malamute',
            250: 'Siberian Husky', 251: 'Affenpinscher', 252: 'Basenji', 253: 'Pug',
            254: 'Leonberg', 255: 'Newfoundland', 256: 'Great Pyrenees', 257: 'Samoyed',
            258: 'Pomeranian', 259: 'Chow Chow', 260: 'Keeshond', 261: 'Brabancon Griffon',
            262: 'Pembroke Welsh Corgi', 263: 'Cardigan Welsh Corgi', 264: 'Toy Poodle',
            265: 'Miniature Poodle', 266: 'Standard Poodle', 267: 'Mexican Hairless'
        }
        
        self.class_labels = {**cat_classes, **dog_classes}
    
    def create_gui(self):
        """Create the professional GUI layout"""
        
        # Header
        header_frame = ttk.Frame(self.root, style='Primary.TFrame', height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = ttk.Label(header_frame, text="🐾 Animal Classifier Pro", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="AI-Powered Cat vs Dog Detection", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(side=tk.LEFT, padx=10, pady=20)
        
        # Main content area
        main_container = ttk.Frame(self.root, style='Background.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls and image list
        left_panel = ttk.Frame(main_container, style='Background.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right panel - Preview and details
        right_panel = ttk.Frame(main_container, style='Background.TFrame', width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Control section
        control_frame = ttk.LabelFrame(left_panel, text="📁 Image Management", 
                                      style='Light.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Button container
        button_container = ttk.Frame(control_frame, style='Light.TFrame')
        button_container.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_container, text="➕ Add Images", 
                  command=self.add_images, style='Primary.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_container, text="🗑️ Clear All", 
                  command=self.clear_images, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="🚀 Analyze All", 
                  command=self.start_analysis, style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(button_container, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Image list section
        list_frame = ttk.LabelFrame(left_panel, text="📋 Selected Images", 
                                   style='Light.TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Image list with scrollbar
        list_container = ttk.Frame(list_frame, style='Light.TFrame')
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_listbox = tk.Listbox(list_container, bg='white', fg=self.colors['dark'],
                                       selectbackground=self.colors['secondary'],
                                       font=('Segoe UI', 9))
        
        list_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, 
                                      command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Results section
        results_frame = ttk.LabelFrame(left_panel, text="📊 Analysis Results", 
                                      style='Light.TFrame')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results treeview
        tree_container = ttk.Frame(results_frame, style='Light.TFrame')
        tree_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('Image', 'True Label', 'Prediction', 'Confidence', 'Status')
        self.results_tree = ttk.Treeview(tree_container, columns=columns, 
                                        show='headings', height=12, 
                                        style='Custom.Treeview')
        
        # Configure columns
        column_widths = {'Image': 150, 'True Label': 100, 'Prediction': 150, 
                        'Confidence': 100, 'Status': 100}
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=column_widths.get(col, 100))
        
        # Treeview scrollbar
        tree_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, 
                                      command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel content
        # Preview section
        preview_frame = ttk.LabelFrame(right_panel, text="🖼️ Image Preview", 
                                      style='Light.TFrame')
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.preview_label = ttk.Label(preview_frame, 
                                      text="Select an image to preview\n\n📷",
                                      background='white',
                                      justify=tk.CENTER,
                                      font=('Segoe UI', 11),
                                      anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Details section
        details_frame = ttk.LabelFrame(right_panel, text="📝 Analysis Details", 
                                      style='Light.TFrame')
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        # Details text widget
        details_container = ttk.Frame(details_frame, style='Light.TFrame')
        details_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.details_text = tk.Text(details_container, bg='white', fg=self.colors['dark'],
                                   font=('Segoe UI', 9), wrap=tk.WORD, height=8,
                                   relief='flat', padx=10, pady=10)
        
        details_scrollbar = ttk.Scrollbar(details_container, orient=tk.VERTICAL,
                                         command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Add images to begin analysis")
        
        status_bar = ttk.Frame(self.root, style='Primary.TFrame', height=25)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        
        status_label = ttk.Label(status_bar, textvariable=self.status_var,
                                background=self.colors['primary'],
                                foreground='white',
                                font=('Segoe UI', 9))
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Bind events
        self.results_tree.bind('<<TreeviewSelect>>', self.show_selected_image)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        
        # Configure tags for results tree
        self.results_tree.tag_configure('Correct', background='#d4edda')
        self.results_tree.tag_configure('Misclassified', background='#f8d7da')
        self.results_tree.tag_configure('Unknown', background='#fff3cd')
    
    def add_images(self):
        """Add images to the list"""
        files = filedialog.askopenfilenames(
            title="Select Animal Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        new_count = 0
        for file_path in files:
            if file_path not in self.image_paths:
                self.image_paths.append(file_path)
                filename = os.path.basename(file_path)
                self.image_listbox.insert(tk.END, filename)
                new_count += 1
        
        if new_count > 0:
            self.status_var.set(f"Added {new_count} new image(s). Total: {len(self.image_paths)}")
        
        # Auto-preview first image if none selected
        if not self.results_tree.selection() and self.image_paths:
            self.image_listbox.selection_set(0)
            self.on_listbox_select(None)
    
    def clear_images(self):
        """Clear all images and results"""
        if self.image_paths and not messagebox.askyesno("Confirm Clear", 
                                                       "Are you sure you want to remove all images and results?"):
            return
        
        self.image_paths.clear()
        self.image_listbox.delete(0, tk.END)
        self.clear_results()
        self.status_var.set("All images and results cleared")
    
    def clear_results(self):
        """Clear results from treeview and details"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        self.details_text.delete(1.0, tk.END)
        self.preview_label.configure(image='', 
                                   text="Select an image to preview\n\n📷")
        self.current_image = None
    
    def on_listbox_select(self, event):
        """Handle listbox selection for quick preview"""
        selection = self.image_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        image_path = self.image_paths[index]
        self.show_image_preview(image_path)
    
    def show_image_preview(self, image_path):
        """Show image preview in the preview panel"""
        try:
            image = Image.open(image_path)
            # Calculate size to fit in preview area while maintaining aspect ratio
            max_size = (300, 300)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo
            
            # Update details
            file_info = f"File: {os.path.basename(image_path)}\n"
            file_info += f"Size: {image.size[0]} x {image.size[1]} pixels\n"
            file_info += f"Format: {image.format if image.format else 'Unknown'}\n"
            file_info += f"Mode: {image.mode}\n\n"
            file_info += "Click 'Analyze All' to classify this image"
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, file_info)
            
        except Exception as e:
            self.preview_label.configure(image='', 
                                       text=f"❌ Error loading preview\n{str(e)}")
    
    def start_analysis(self):
        """Start analysis in a separate thread"""
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please add images first!")
            return
        
        if self.is_analyzing:
            messagebox.showinfo("Analysis Running", "Analysis is already in progress!")
            return
        
        # Disable analyze button during analysis
        self.is_analyzing = True
        self.status_var.set("Starting analysis...")
        
        # Start analysis in separate thread
        thread = threading.Thread(target=self.analyze_images)
        thread.daemon = True
        thread.start()
    
    def analyze_images(self):
        """Analyze all loaded images"""
        self.clear_results()
        total_images = len(self.image_paths)
        misclassified_count = 0
        analysis_start = datetime.now()
        
        # Update progress
        self.root.after(0, lambda: self.progress.configure(maximum=total_images))
        
        for i, image_path in enumerate(self.image_paths):
            if not self.is_analyzing:  # Allow cancellation
                break
                
            filename = os.path.basename(image_path)
            self.status_var.set(f"Analyzing {i+1}/{total_images}: {filename}")
            
            # Classify image
            predicted_class, confidence, image = self.classify_image(image_path)
            
            if predicted_class:
                true_label = "Dog"  # Assuming all are dogs for this use case
                
                is_dog_prediction = self.is_dog_breed(predicted_class)
                is_cat_prediction = self.is_cat_breed(predicted_class)
                
                if is_dog_prediction:
                    status = "Correct"
                    status_icon = "✅"
                elif is_cat_prediction:
                    status = "Misclassified"
                    status_icon = "❌"
                    misclassified_count += 1
                else:
                    status = "Unknown"
                    status_icon = "❓"
                
                # Add to results tree
                self.root.after(0, self.add_result_row, (
                    filename, true_label, predicted_class, 
                    f"{confidence:.2%}", f"{status_icon} {status}"
                ), status)
            
            # Update progress
            self.root.after(0, lambda val=i+1: self.progress.configure(value=val))
        
        # Analysis complete
        analysis_time = (datetime.now() - analysis_start).total_seconds()
        
        self.root.after(0, self.analysis_complete, total_images, 
                       misclassified_count, analysis_time)
    
    def analysis_complete(self, total, misclassified, time_taken):
        """Handle analysis completion"""
        self.is_analyzing = False
        self.progress.configure(value=0)
        
        accuracy = (total - misclassified) / total if total > 0 else 0
        
        summary = f"Analysis Complete!\n\n"
        summary += f"📊 Summary:\n"
        summary += f"• Total images analyzed: {total}\n"
        summary += f"• Correctly classified: {total - misclassified}\n"
        summary += f"• Misclassified: {misclassified}\n"
        summary += f"• Accuracy: {accuracy:.1%}\n"
        summary += f"• Time taken: {time_taken:.2f} seconds\n\n"
        
        if misclassified == 0:
            summary += "🎉 Perfect accuracy! All dogs correctly identified!"
        elif accuracy >= 0.8:
            summary += "👍 Good results! The model performed well."
        else:
            summary += "💡 Consider using more training data or a different model."
        
        self.status_var.set(f"Analysis complete - Accuracy: {accuracy:.1%}")
        
        # Show summary in details
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, summary)
        
        messagebox.showinfo("Analysis Complete", summary)
    
    def add_result_row(self, values, status):
        """Add a row to results treeview"""
        self.results_tree.insert('', tk.END, values=values, tags=(status,))
    
    def classify_image(self, image_path):
        """Classify a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 3)  # Get top 3 predictions
            
            class_id = top_catid[0].item()
            class_name = self.class_labels.get(class_id, f"Unknown (ID: {class_id})")
            confidence = top_prob[0].item()
            
            return class_name, confidence, image
        
        except Exception as e:
            print(f"Error classifying {image_path}: {e}")
            return f"Error: {str(e)}", 0.0, None
    
    def is_dog_breed(self, class_name):
        """Check if prediction is a dog breed"""
        dog_indicators = ['dog', 'hound', 'terrier', 'retriever', 'spaniel', 
                         'sheepdog', 'bulldog', 'poodle', 'mastiff', 'setter',
                         'pinscher', 'wolfhound', 'greyhound', 'whippet']
        return any(indicator in class_name.lower() for indicator in dog_indicators)
    
    def is_cat_breed(self, class_name):
        """Check if prediction is a cat breed"""
        cat_indicators = ['cat', 'tabby', 'siamese', 'persian', 'egyptian']
        return any(indicator in class_name.lower() for indicator in cat_indicators)
    
    def show_selected_image(self, event):
        """Show the selected analyzed image with results"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.results_tree.item(item, 'values')
        filename = values[0]
        
        # Find the image path
        image_path = None
        for path in self.image_paths:
            if os.path.basename(path) == filename:
                image_path = path
                break
        
        if image_path:
            try:
                image = Image.open(image_path)
                max_size = (300, 300)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
                
                # Show detailed results
                details = f"📄 File: {values[0]}\n"
                details += f"🏷️ True Label: {values[1]}\n"
                details += f"🤖 AI Prediction: {values[2]}\n"
                details += f"📈 Confidence: {values[3]}\n"
                details += f"📊 Status: {values[4]}\n\n"
                
                # Add interpretation
                if "Correct" in values[4]:
                    details += "✅ The AI correctly identified this as a dog!"
                elif "Misclassified" in values[4]:
                    details += "❌ The AI misclassified this dog as a cat."
                else:
                    details += "❓ The AI prediction is uncertain."
                
                self.details_text.delete(1.0, tk.END)
                self.details_text.insert(1.0, details)
                
            except Exception as e:
                self.preview_label.configure(image='', 
                                           text=f"Error loading image\n{str(e)}")

def main():
    root = tk.Tk()
    app = ProfessionalCatDogClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()