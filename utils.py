import numpy as np
import cv2 # Sửa lỗi import
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Định nghĩa Anchor Boxes (ví dụ, 3 anchors)
# Kích thước được chuẩn hóa tương đối với kích thước grid cell
# Ví dụ: nếu grid_size=8, ảnh 64x64, thì 1 cell là 8x8 pixels
# Các giá trị này (width, height) nên được tính toán từ dataset của bạn bằng K-Means
# Ví dụ này giả định kích thước tương đối so với grid cell (ví dụ: 1.0 nghĩa là bằng kích thước cell)
ANCHORS = np.array([
    [0.5, 0.5],  # Anchor nhỏ, vuông (ví dụ: 4x4 pixels trên grid 8x8)
    [1.0, 1.0],  # Anchor trung bình, vuông (ví dụ: 8x8 pixels)
    [2.0, 2.0]   # Anchor lớn, vuông (ví dụ: 16x16 pixels)
], dtype=np.float32)
NUM_ANCHORS = len(ANCHORS)

def iou(box1, box2):
    """Tính IoU giữa hai box (hoặc một box và nhiều box)
    Box format: [x_center, y_center, width, height]
    """
    # Chuyển sang [x_min, y_min, x_max, y_max]
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxs = box1_xy + box1_wh / 2.

    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxs = box2_xy + box2_wh / 2.

    # Tính diện tích giao nhau
    intersect_mins = np.maximum(box1_mins, box2_mins)
    intersect_maxs = np.minimum(box1_maxs, box2_maxs)
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Tính diện tích riêng
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]

    # Tính IoU
    union_area = box1_area + box2_area - intersect_area
    iou_val = intersect_area / (union_area + 1e-6) # Thêm epsilon tránh chia cho 0

    return iou_val

def non_max_suppression(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.5):
    """Applies Non-Maximum Suppression (NMS) to bounding boxes."""
    # Filter out boxes below the score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    if len(boxes) == 0:
        return [], [], []

    # Convert boxes to [y_min, x_min, y_max, x_max] for tf.image.non_max_suppression
    # Input boxes are [x_min, y_min, x_max, y_max]
    boxes_tf = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

    selected_indices = tf.image.non_max_suppression(
        boxes=boxes_tf,
        scores=scores,
        max_output_size=100,  # Limit the number of final boxes
        iou_threshold=iou_threshold
    )

    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(scores, selected_indices).numpy()
    selected_classes = tf.gather(classes, selected_indices).numpy()

    return selected_boxes, selected_scores, selected_classes

def process_predictions(predictions, grid_size=8, confidence_threshold=0.5, nms_threshold=0.5, num_classes=1, anchors=ANCHORS, num_anchors=NUM_ANCHORS):
    """Process model predictions with anchors to get bounding boxes"""
    # predictions shape: [batch, grid_size, grid_size, num_anchors, 5+num_classes]
    batch_size = tf.shape(predictions)[0]
    all_boxes = []
    all_scores = []
    all_classes = []

    # Tạo grid cell indices để decode tọa độ
    grid_y = tf.tile(tf.range(0, grid_size), [grid_size])
    grid_x = tf.keras.backend.repeat_elements(tf.range(0, grid_size), grid_size, axis=0)
    grid_xy = tf.stack([grid_x, grid_y], axis=-1) # Shape: [S*S, 2]
    grid_xy = tf.reshape(grid_xy, [1, grid_size, grid_size, 1, 2]) # Shape: [1, S, S, 1, 2]
    grid_xy = tf.cast(grid_xy, tf.float32)

    # Reshape anchors để broadcast: [1, 1, 1, num_anchors, 2]
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    # Decode predictions
    pred_xy = predictions[..., 0:2] # Sigmoid đã được áp dụng trong model
    pred_wh_rel = predictions[..., 2:4] # Output linear/log từ model
    pred_obj = predictions[..., 4:5] # Sigmoid đã được áp dụng
    pred_class = predictions[..., 5:] # Sigmoid/Softmax đã được áp dụng

    # Decode xy: (sigmoid(pred_xy) + grid_xy) / grid_size
    pred_xy_abs = (pred_xy + grid_xy) / tf.cast(grid_size, tf.float32)
    # Decode wh: exp(pred_wh_rel) * anchor_wh / grid_size
    pred_wh_abs = tf.exp(pred_wh_rel) * anchors_tensor / tf.cast(grid_size, tf.float32)

    # Ghép lại thành box tuyệt đối [x_center, y_center, width, height] (normalized 0-1)
    pred_box_abs = tf.concat([pred_xy_abs, pred_wh_abs], axis=-1)

    # Tính final scores = objectness * class_probability
    # Giả sử dùng sigmoid cho class (lấy score cao nhất)
    pred_class_max_score = tf.reduce_max(pred_class, axis=-1, keepdims=True) # Lấy score cao nhất
    pred_class_max_id = tf.cast(tf.argmax(pred_class, axis=-1), tf.float32) # Lấy class id tương ứng
    pred_class_max_id = tf.expand_dims(pred_class_max_id, axis=-1) # Thêm chiều cuối

    final_scores = pred_obj * pred_class_max_score # Shape: [batch, S, S, num_anchors, 1]

    # Reshape để dễ xử lý và lọc
    # Shape: [batch, S * S * num_anchors, 4]
    pred_box_flat = tf.reshape(pred_box_abs, [batch_size, -1, 4])
    # Shape: [batch, S * S * num_anchors, 1]
    final_scores_flat = tf.reshape(final_scores, [batch_size, -1, 1])
    # Shape: [batch, S * S * num_anchors, 1] - Đã sửa lỗi
    pred_class_flat = tf.reshape(pred_class_max_id, [batch_size, -1, 1])

    for b in range(batch_size):
        boxes_b = pred_box_flat[b] # Shape: [N, 4] (N = S*S*num_anchors)
        scores_b = final_scores_flat[b, ..., 0] # Shape: [N]
        classes_b = pred_class_flat[b, ..., 0] # Shape: [N]

        # Chuyển boxes sang format [x_min, y_min, x_max, y_max]
        boxes_xy = boxes_b[..., :2]
        boxes_wh = boxes_b[..., 2:4]
        boxes_min = boxes_xy - boxes_wh / 2.0
        boxes_max = boxes_xy + boxes_wh / 2.0
        boxes_nms_fmt = tf.concat([boxes_min, boxes_max], axis=-1) # [x_min, y_min, x_max, y_max]

        # Apply Non-Maximum Suppression
        final_boxes, final_scores, final_classes = non_max_suppression(
            boxes=boxes_nms_fmt.numpy(), # Chuyển sang numpy
            scores=scores_b.numpy(),
            classes=classes_b.numpy(),
            iou_threshold=nms_threshold,
            score_threshold=confidence_threshold
        )

        all_boxes.append(final_boxes)
        all_scores.append(final_scores)
        all_classes.append(final_classes)

    return all_boxes, all_scores, all_classes

def draw_boxes(image, boxes, scores, class_ids, class_names=None, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on an image."""
    img_copy = image.copy() # Avoid modifying original image
    h, w, _ = img_copy.shape

    for i, box in enumerate(boxes):
        score = scores[i]
        class_id = class_ids[i]

        # Box coordinates are normalized [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = box

        # Convert normalized coordinates to pixel coordinates
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        # Draw the box
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)

        # Draw label
        label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Put label background
        cv2.rectangle(
            img_copy,
            (x_min, y_min - label_height - baseline),
            (x_min + label_width, y_min),
            color,
            cv2.FILLED
        )
        # Put label text
        cv2.putText(
            img_copy,
            label,
            (x_min, y_min - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            1,
            cv2.LINE_AA
        )
    return img_copy

def plot_detection_results(images, all_boxes, all_scores, all_classes, class_names=None, figsize=(12, 12)):
    """Plot detection results for multiple images using matplotlib"""
    num_images = len(images)
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() # Flatten to easily iterate

    for i in range(num_images):
        ax = axes[i]
        img = images[i] # Assuming images are normalized [0,1] RGB
        boxes = all_boxes[i]
        scores = all_scores[i]
        classes = all_classes[i]

        # Display image
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Image {i}')

        # Draw boxes using matplotlib patches
        for j, box in enumerate(boxes):
            score = scores[j]
            class_id = int(classes[j])
            x_min, y_min, x_max, y_max = box

            # Convert normalized coordinates to absolute coordinates for patches
            img_h, img_w = img.shape[:2] # Get image dimensions (assuming normalized image)
            abs_x_min = x_min * img_w
            abs_y_min = y_min * img_h
            abs_width = (x_max - x_min) * img_w
            abs_height = (y_max - y_min) * img_h

            # Create a Rectangle patch
            rect = patches.Rectangle(
                (abs_x_min, abs_y_min), abs_width, abs_height,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add label text
            label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
            ax.text(
                abs_x_min, abs_y_min - 2, label,
                color='white', fontsize=8,
                bbox=dict(facecolor='red', alpha=0.5, pad=0)
            )

    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig # Return the figure object

class YOLODatasetFromPaths(Sequence):
    """YOLO dataset from text file with list of images (YOLOv5 format)"""
    def __init__(self, txt_path, img_size=64, grid_size=8, batch_size=16, num_classes=1, class_names=None, anchors=ANCHORS, num_anchors=NUM_ANCHORS, augment=True, verbose=True): # Thêm anchors, num_anchors
        self.txt_path = txt_path
        self.img_size = img_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.anchors = anchors # Lưu anchors
        self.num_anchors = num_anchors # Lưu số lượng anchors
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

            # Output: [S, S, num_anchors, 5+num_classes] for each grid cell: [x, y, w, h, objectness, class_probs]
            y = np.zeros((S, S, self.num_anchors, 5 + self.num_classes))

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

                # Tính IoU giữa ground truth box và các anchor boxes
                gt_box_shape = np.array([0, 0, width * S, height * S])
                anchor_shapes = np.concatenate([np.zeros((self.num_anchors, 2)), self.anchors], axis=-1)
                ious = iou(gt_box_shape, anchor_shapes)
                best_anchor_index = np.argmax(ious)
                # Gán thông tin vào anchor chịu trách nhiệm
                if y[grid_y, grid_x, best_anchor_index, 4] == 0:
                    x_cell = x_center * S - grid_x    # Sử dụng x_center*S thay grid_x_float 
                    y_cell = y_center * S - grid_y    # Sử dụng y_center*S thay grid_y_float
                    anchor_w, anchor_h = self.anchors[best_anchor_index]
                    w_rel = np.log((width * S / anchor_w) + 1e-9)
                    h_rel = np.log((height * S / anchor_h) + 1e-9)
                    y[grid_y, grid_x, best_anchor_index, 0] = x_cell
                    y[grid_y, grid_x, best_anchor_index, 1] = y_cell
                    y[grid_y, grid_x, best_anchor_index, 2] = w_rel
                    y[grid_y, grid_x, best_anchor_index, 3] = h_rel
                    y[grid_y, grid_x, best_anchor_index, 4] = 1.0
                    y[grid_y, grid_x, best_anchor_index, 5 + class_id] = 1.0

            batch_x.append(img)
            batch_y.append(y)

        # Nếu batch rỗng, tạo dummy data
        if len(batch_x) == 0:
            dummy_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            dummy_y = np.zeros((self.grid_size, self.grid_size, self.num_anchors, 5 + self.num_classes), dtype=np.float32)
            batch_x = [dummy_img]
            batch_y = [dummy_y]
            print("Warning: Empty batch, using dummy data")

        return np.array(batch_x), np.array(batch_y)

def get_data_loaders(data_path, img_size=64, grid_size=8, batch_size=16, anchors=ANCHORS, num_anchors=NUM_ANCHORS):
    """Create data loaders for training and validation"""
    print(f"Tạo data loader với kích thước ảnh {img_size}x{img_size}, grid_size={grid_size}, num_anchors={num_anchors}")
    train_dataset = YOLODataset(
        os.path.join(data_path, 'train'),
        img_size=img_size,
        grid_size=grid_size,
        batch_size=batch_size,
        anchors=anchors,
        num_anchors=num_anchors,
        augment=True
    )
    val_dataset = YOLODataset(
        os.path.join(data_path, 'val'),
        img_size=img_size,
        grid_size=grid_size,
        batch_size=batch_size,
        anchors=anchors,
        num_anchors=num_anchors,
        augment=False
    )
    # Lấy num_classes từ train_dataset sau khi khởi tạo
    num_classes = train_dataset.num_classes
    return train_dataset, val_dataset, num_classes