import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2

class YOLODataset(Sequence):
    def __init__(self, dataset_path, img_size=64, grid_size=8, batch_size=16, augment=True, verbose=True):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.augment = augment
        self.verbose = verbose
        
        # Kiểm tra và điều chỉnh grid_size nếu cần
        if self.img_size % self.grid_size != 0:
            original_grid = self.grid_size
            # Tìm grid_size phù hợp (chia hết)
            for gs in [8, 4, 16, 32]:
                if self.img_size % gs == 0:
                    self.grid_size = gs
                    break
            print(f"CẢNH BÁO: Điều chỉnh grid_size từ {original_grid} thành {self.grid_size} để phù hợp với img_size={self.img_size}")
        
        # YOLO dataset structure: dataset_path/images and dataset_path/labels
        self.img_dir = os.path.join(dataset_path, 'images')
        self.label_dir = os.path.join(dataset_path, 'labels')
        
        # Get image files that have corresponding labels
        self.img_files = []
        for img_file in sorted(os.listdir(self.img_dir)):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(self.label_dir, base_name + '.txt')
            
            if os.path.exists(label_file):
                self.img_files.append(os.path.join(self.img_dir, img_file))
        
        # Read class names if available
        self.classes_file = os.path.join(dataset_path, 'classes.txt')
        self.class_names = []
        if os.path.exists(self.classes_file):
            with open(self.classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        
        self.num_classes = len(self.class_names) if self.class_names else 0
        print(f"Found {len(self.img_files)} images with labels")
        print(f"Number of classes: {self.num_classes}")
        
        # Kiểm tra kích thước ảnh đầu tiên để cảnh báo về resize
        if len(self.img_files) > 0 and self.verbose:
            first_img = cv2.imread(self.img_files[0])
            if first_img is not None:
                orig_h, orig_w = first_img.shape[:2]
                if orig_h != self.img_size or orig_w != self.img_size:
                    print(f"THÔNG BÁO: Ảnh gốc có kích thước {orig_w}x{orig_h} sẽ được tự động resize về {self.img_size}x{self.img_size}")
    
    def __len__(self):
        return len(self.img_files) // self.batch_size
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.img_files))):
            img_path = self.img_files[i]
            label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            
            # Kiểm tra và thông báo khi resize ảnh không đúng kích thước (chỉ trong lần đầu)
            if i == idx * self.batch_size and self.verbose:
                orig_h, orig_w = img.shape[:2]
                if orig_h != self.img_size or orig_w != self.img_size:
                    print(f"Resizing image from {orig_w}x{orig_h} to {self.img_size}x{self.img_size}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            
            # Load labels (YOLO format: class x_center y_center width height)
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to xmin, ymin, xmax, ymax format
                            x_min = max(0, x_center - width/2)
                            y_min = max(0, y_center - height/2)
                            x_max = min(1, x_center + width/2)
                            y_max = min(1, y_center + height/2)
                            
                            labels.append([class_id, x_min, y_min, x_max, y_max])
            
            # Prepare output tensor for detection network
            # For this example, we'll use a simple grid-based representation
            # Grid cells: S×S
            S = self.grid_size
            grid_cell_size = self.img_size // S
            
            # Output: [S, S, 5+num_classes] for each grid cell: [x, y, w, h, objectness, class_probs]
            y = np.zeros((S, S, 5 + self.num_classes))
            
            for label in labels:
                class_id, x_min, y_min, x_max, y_max = label
                
                # Calculate box center and dimensions
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Find grid cell responsible for this box
                grid_x = int(x_center * S)
                grid_y = int(y_center * S)
                
                # Đảm bảo grid_x và grid_y nằm trong giới hạn hợp lệ
                grid_x = min(max(0, grid_x), S-1)
                grid_y = min(max(0, grid_y), S-1)
                
                # Convert coordinates relative to grid cell
                x_cell = x_center * S - grid_x
                y_cell = y_center * S - grid_y
                w_cell = width * S
                h_cell = height * S
                
                # Only one prediction per grid cell for simplicity
                if y[grid_y, grid_x, 4] == 0:  # If no object already assigned
                    y[grid_y, grid_x, 0] = x_cell
                    y[grid_y, grid_x, 1] = y_cell
                    y[grid_y, grid_x, 2] = w_cell
                    y[grid_y, grid_x, 3] = h_cell
                    y[grid_y, grid_x, 4] = 1.0  # objectness
                    y[grid_y, grid_x, 5 + class_id] = 1.0  # class probability
            
            batch_x.append(img)
            batch_y.append(y)
        
        return np.array(batch_x), np.array(batch_y)

class YOLODatasetFromPaths(Sequence):
    """YOLO dataset from text file with list of images (YOLOv5 format)"""
    def __init__(self, txt_path, img_size=64, grid_size=8, batch_size=16, num_classes=1, class_names=None, augment=True, verbose=True):
        self.txt_path = txt_path
        self.img_size = img_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.augment = augment
        self.verbose = verbose
        self.resize_warning_shown = False  # Flag để kiểm soát việc hiển thị thông báo resize
        
        # Kiểm tra và điều chỉnh grid_size nếu cần
        if self.img_size % self.grid_size != 0:
            original_grid = self.grid_size
            # Tìm grid_size phù hợp (chia hết)
            for gs in [8, 4, 16, 32]:
                if self.img_size % gs == 0:
                    self.grid_size = gs
                    break
            print(f"CẢNH BÁO: Điều chỉnh grid_size từ {original_grid} thành {self.grid_size} để phù hợp với img_size={self.img_size}")
        
        # Đọc danh sách đường dẫn ảnh từ file txt
        self.img_files = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                self.img_files = [line.strip() for line in f.readlines() if line.strip()]
        
        # In thông tin
        print(f"Loaded dataset from {txt_path} with {len(self.img_files)} images")
        print(f"Number of classes: {self.num_classes}")
        
        # Kiểm tra kích thước ảnh đầu tiên để cảnh báo về resize
        if len(self.img_files) > 0 and self.verbose:
            first_img = cv2.imread(self.img_files[0])
            if first_img is not None:
                orig_h, orig_w = first_img.shape[:2]
                if orig_h != self.img_size or orig_w != self.img_size:
                    print(f"THÔNG BÁO: Ảnh gốc có kích thước {orig_w}x{orig_h} sẽ được tự động resize về {self.img_size}x{self.img_size}")
    
    def __len__(self):
        return max(1, len(self.img_files) // self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.img_files))
        
        for i in range(start_idx, end_idx):
            img_path = self.img_files[i]
            
            # Tìm file label tương ứng (thay .jpg/.png/... thành .txt)
            label_path = img_path.replace('images', 'labels')
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                label_path = label_path.replace(ext, '.txt')
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Kiểm tra và thông báo khi resize ảnh không đúng kích thước (chỉ trong lần đầu)
            if i == start_idx and not self.resize_warning_shown:
                orig_h, orig_w = img.shape[:2]
                if orig_h != self.img_size or orig_w != self.img_size:
                    print(f"Resizing image from {orig_w}x{orig_h} to {self.img_size}x{self.img_size}")
                    self.resize_warning_shown = True
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            
            # Load labels (YOLO format: class x_center y_center width height)
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(float(parts[0]))
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to xmin, ymin, xmax, ymax format
                            x_min = max(0, x_center - width/2)
                            y_min = max(0, y_center - height/2)
                            x_max = min(1, x_center + width/2)
                            y_max = min(1, y_center + height/2)
                            
                            # Kiểm tra class_id có hợp lệ không
                            if 0 <= class_id < self.num_classes:
                                labels.append([class_id, x_min, y_min, x_max, y_max])
                            else:
                                print(f"Warning: Invalid class_id {class_id} in {label_path}")
            
            # Prepare output tensor for detection network
            # For this example, we'll use a simple grid-based representation
            # Grid cells: S×S
            S = self.grid_size
            grid_cell_size = self.img_size // S
            
            # Output: [S, S, 5+num_classes] for each grid cell: [x, y, w, h, objectness, class_probs]
            y = np.zeros((S, S, 5 + self.num_classes))
            
            for label in labels:
                class_id, x_min, y_min, x_max, y_max = label
                
                # Calculate box center and dimensions
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Find grid cell responsible for this box
                grid_x = int(x_center * S)
                grid_y = int(y_center * S)
                
                # Đảm bảo grid_x và grid_y nằm trong giới hạn hợp lệ
                grid_x = min(max(0, grid_x), S-1)
                grid_y = min(max(0, grid_y), S-1)
                
                # Convert coordinates relative to grid cell
                x_cell = x_center * S - grid_x
                y_cell = y_center * S - grid_y
                w_cell = width * S
                h_cell = height * S
                
                # Only one prediction per grid cell for simplicity
                if y[grid_y, grid_x, 4] == 0:  # If no object already assigned
                    y[grid_y, grid_x, 0] = x_cell
                    y[grid_y, grid_x, 1] = y_cell
                    y[grid_y, grid_x, 2] = w_cell
                    y[grid_y, grid_x, 3] = h_cell
                    y[grid_y, grid_x, 4] = 1.0  # objectness
                    y[grid_y, grid_x, 5 + class_id] = 1.0  # class probability
            
            batch_x.append(img)
            batch_y.append(y)
        
        # Nếu batch rỗng, tạo dummy data
        if len(batch_x) == 0:
            dummy_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            dummy_y = np.zeros((self.grid_size, self.grid_size, 5 + self.num_classes), dtype=np.float32)
            batch_x = [dummy_img]
            batch_y = [dummy_y]
            print("Warning: Empty batch, using dummy data")
        
        return np.array(batch_x), np.array(batch_y)

def get_data_loaders(data_path, img_size=64, grid_size=8, batch_size=16):
    """Create data loaders for training and validation"""
    print(f"Tạo data loader với kích thước ảnh {img_size}x{img_size}, grid_size={grid_size}")
    
    train_dataset = YOLODataset(
        os.path.join(data_path, 'train'),
        img_size=img_size,
        grid_size=grid_size,
        batch_size=batch_size,
        augment=True
    )
    
    val_dataset = YOLODataset(
        os.path.join(data_path, 'val'),
        img_size=img_size,
        grid_size=grid_size,
        batch_size=batch_size,
        augment=False
    )
    
    return train_dataset, val_dataset, train_dataset.num_classes 