import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from tqdm import tqdm # Import tqdm for progress bar

from dataset import get_data_loaders, YOLODatasetFromPaths
from detection_model import build_detection_model
from utils import process_predictions, draw_boxes, plot_detection_results
from utils import ANCHORS, NUM_ANCHORS, process_predictions, draw_boxes, plot_detection_results, iou

class DetectionMetricsCallback(Callback):
    """Callback để tính toán và hiển thị AP và AR sau mỗi epoch trên toàn bộ validation set"""
    def __init__(self, validation_data, grid_size, num_classes, anchors=ANCHORS, num_anchors=NUM_ANCHORS, iou_threshold=0.5, confidence_threshold=0.1, nms_threshold=0.5, frequency=1):
        super().__init__()
        self.validation_data = validation_data
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.anchors = tf.constant(anchors, dtype=tf.float32) # Lưu anchors
        self.num_anchors = num_anchors # Lưu num_anchors
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.frequency = frequency
        self.ap_history = []
        self.ar_history = []
        self.last_ap = 0.0
        self.last_ar = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Chỉ tính toán nếu epoch chia hết cho frequency
        if (epoch + 1) % self.frequency == 0:
            print(f"\nCalculating validation metrics for epoch {epoch+1}...")
            all_true_boxes = []
            all_true_classes = []
            all_pred_boxes = []
            all_pred_scores = []
            all_pred_classes = []

            # Lặp qua toàn bộ validation dataset
            for batch_x, batch_y in tqdm(self.validation_data, desc="Validation Metrics"):
                # Dự đoán trên batch hiện tại
                predictions = self.model.predict(batch_x, verbose=0)

                # Xử lý dự đoán cho batch này (áp dụng confidence threshold và NMS)
                # Truyền anchors và num_anchors vào process_predictions
                batch_pred_boxes, batch_pred_scores, batch_pred_classes = process_predictions(
                    predictions,
                    grid_size=self.grid_size,
                    confidence_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold,
                    num_classes=self.num_classes,
                    anchors=self.anchors.numpy(), # Truyền anchors (numpy array)
                    num_anchors=self.num_anchors  # Truyền num_anchors
                )

                # Trích xuất ground truth cho batch này (đã cập nhật)
                batch_true_boxes, batch_true_classes = self._extract_true_boxes_batch(batch_y)

                # Lưu kết quả của batch
                all_true_boxes.extend(batch_true_boxes)
                all_true_classes.extend(batch_true_classes)
                all_pred_boxes.extend(batch_pred_boxes)
                all_pred_scores.extend(batch_pred_scores)
                all_pred_classes.extend(batch_pred_classes)

            # Tính toán AP và AR tổng thể trên toàn bộ dataset
            ap, ar = self.calculate_mean_ap_ar(
                all_true_boxes, all_true_classes,
                all_pred_boxes, all_pred_scores, all_pred_classes
            )

            # Lưu giá trị mới nhất
            self.last_ap = ap
            self.last_ar = ar

            # Thêm vào lịch sử
            self.ap_history.append(ap)
            self.ar_history.append(ar)

            # Thêm vào logs
            logs['val_AP'] = ap
            logs['val_AR'] = ar
            print(f"Epoch {epoch+1}: val_AP: {ap:.4f} - val_AR: {ar:.4f}")
        else:
             # Sử dụng lại giá trị gần nhất cho các epoch không tính toán
             logs['val_AP'] = self.last_ap
             logs['val_AR'] = self.last_ar

    def _extract_true_boxes_batch(self, batch_y):
        """Trích xuất ground truth boxes và classes từ batch target với anchor"""
        batch_true_boxes = []
        batch_true_classes = []
        S = self.grid_size
        # Chuyển anchors sang numpy để dễ truy cập
        anchors_np = self.anchors.numpy()

        for y_true in batch_y: # y_true shape: [S, S, num_anchors, 5+C]
            true_boxes_img = []
            true_classes_img = []
            for row in range(S):
                for col in range(S):
                    for anchor_idx in range(self.num_anchors):
                        if y_true[row, col, anchor_idx, 4] > 0.5:  # Nếu anchor này chịu trách nhiệm (objectness > 0.5)
                            x_cell = y_true[row, col, anchor_idx, 0]
                            y_cell = y_true[row, col, anchor_idx, 1]
                            w_rel = y_true[row, col, anchor_idx, 2] # log encoded
                            h_rel = y_true[row, col, anchor_idx, 3] # log encoded
                            class_id = np.argmax(y_true[row, col, anchor_idx, 5:])

                            # Decode coordinates (relative to image)
                            x_center = (col + x_cell) / S
                            y_center = (row + y_cell) / S

                            # Decode dimensions (relative to image)
                            anchor_w, anchor_h = anchors_np[anchor_idx]
                            w = (np.exp(w_rel) * anchor_w) / S
                            h = (np.exp(h_rel) * anchor_h) / S

                            # Convert to [x_min, y_min, x_max, y_max]
                            x_min = max(0.0, x_center - w/2)
                            y_min = max(0.0, y_center - h/2)
                            x_max = min(1.0, x_center + w/2)
                            y_max = min(1.0, y_center + h/2)

                            true_boxes_img.append([x_min, y_min, x_max, y_max])
                            true_classes_img.append(class_id)

            batch_true_boxes.append(np.array(true_boxes_img) if true_boxes_img else np.zeros((0, 4)))
            batch_true_classes.append(np.array(true_classes_img, dtype=int) if true_classes_img else np.array([], dtype=int))
        return batch_true_boxes, batch_true_classes

    def calculate_mean_ap_ar(self, all_true_boxes, all_true_classes, all_pred_boxes, all_pred_scores, all_pred_classes):
        """Tính toán mAP và mAR trên toàn bộ dataset"""
        # ... (Logic tính AP/AR giữ nguyên như trước, vì nó hoạt động trên box đã decode) ...
        # Chỉ cần đảm bảo đầu vào (all_true_boxes, all_pred_boxes, ...) là đúng định dạng
        all_scores_list = []
        all_tp_fp_list = [] # 1 for TP, 0 for FP
        total_num_gt = 0
        total_matched_gt = 0

        # Lặp qua từng ảnh trong dataset
        for i in range(len(all_true_boxes)):
            true_boxes = all_true_boxes[i]
            true_classes = all_true_classes[i]
            pred_boxes = all_pred_boxes[i]
            pred_scores = all_pred_scores[i]
            pred_classes = all_pred_classes[i]

            num_gt = len(true_boxes)
            total_num_gt += num_gt

            if num_gt == 0:
                # Nếu không có GT, tất cả dự đoán là FP
                for score in pred_scores:
                    all_scores_list.append(score)
                    all_tp_fp_list.append(0) # FP
                continue

            if len(pred_boxes) == 0:
                # Nếu không có dự đoán, không có TP/FP nào được thêm
                continue

            gt_used = np.zeros(num_gt, dtype=bool)

            # Sắp xếp dự đoán theo điểm số giảm dần (đã được thực hiện trong process_predictions? Kiểm tra lại)
            # Nếu process_predictions chưa sắp xếp, cần sắp xếp ở đây:
            # order = np.argsort(-pred_scores)
            # pred_boxes = pred_boxes[order]
            # pred_scores = pred_scores[order]
            # pred_classes = pred_classes[order]

            # Tính IoU matrix (pred_boxes vs true_boxes)
            # Format: [x_min, y_min, x_max, y_max]
            iou_matrix = self.calculate_iou_matrix(pred_boxes, true_boxes)

            # Xác định TP/FP cho từng dự đoán
            for j in range(len(pred_boxes)):
                pred_class = int(pred_classes[j])
                pred_score = pred_scores[j]

                # Tìm các GT cùng lớp
                gt_same_class_indices = np.where(true_classes == pred_class)[0]

                is_tp = False
                if len(gt_same_class_indices) > 0:
                    ious = iou_matrix[j, gt_same_class_indices]
                    best_match_local_idx = np.argmax(ious)
                    best_iou = ious[best_match_local_idx]
                    best_match_gt_idx = gt_same_class_indices[best_match_local_idx]

                    if best_iou >= self.iou_threshold and not gt_used[best_match_gt_idx]:
                        is_tp = True
                        gt_used[best_match_gt_idx] = True

                all_scores_list.append(pred_score)
                all_tp_fp_list.append(1 if is_tp else 0)

            total_matched_gt += np.sum(gt_used)

        # Tính toán AP tổng thể
        if not all_scores_list: # Nếu không có dự đoán nào trên toàn bộ dataset
             return 0.0, 0.0

        # Sắp xếp dựa trên scores
        scores_array = np.array(all_scores_list)
        tp_fp_array = np.array(all_tp_fp_list)

        order = np.argsort(-scores_array)
        tp_fp_array = tp_fp_array[order]

        cum_tp = np.cumsum(tp_fp_array)
        cum_fp = np.cumsum(1 - tp_fp_array)

        recall = cum_tp / total_num_gt if total_num_gt > 0 else np.zeros_like(cum_tp, dtype=float)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9) # Thêm epsilon để tránh chia cho 0

        ap = self.calculate_ap_from_precision_recall(precision, recall)
        ar = total_matched_gt / total_num_gt if total_num_gt > 0 else 0.0

        return ap, ar

    def calculate_iou_matrix(self, boxes1, boxes2):
        """Tính ma trận IoU giữa hai tập hợp boxes [x_min, y_min, x_max, y_max]"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))

        # Mở rộng boxes để tính toán broadcast
        boxes1 = np.expand_dims(boxes1, axis=1) # Shape: [N, 1, 4]
        boxes2 = np.expand_dims(boxes2, axis=0) # Shape: [1, M, 4]

        # Tính toán phần giao nhau (intersection)
        intersect_mins = np.maximum(boxes1[..., :2], boxes2[..., :2])
        intersect_maxs = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        # Tính toán diện tích của từng box
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # Tính toán phần hợp (union)
        union_area = area1 + area2 - intersect_area

        # Tính IoU, tránh chia cho 0
        iou_matrix = intersect_area / (union_area + 1e-9)

        return iou_matrix

    def calculate_ap_from_precision_recall(self, precision, recall):
        """Tính AP từ precision và recall bằng phương pháp AUC (Average Precision)"""
        # Thêm điểm đầu và cuối để đảm bảo bao phủ toàn bộ recall range [0, 1]
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Làm mịn precision (đảm bảo precision không giảm khi recall tăng)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

        # Tìm các điểm mà recall thay đổi (để tính diện tích các hình chữ nhật)
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Tính diện tích dưới đường cong precision-recall (AUC)
        # Sum( (recall[i+1] - recall[i]) * precision[i+1] )
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

        return ap

def save_model_with_info(model, filepath, info):
    """Save the Keras model and an accompanying JSON file with metadata."""
    model.save(filepath)
    info_filepath = filepath.replace('.keras', '_info.json')
    with open(info_filepath, 'w') as f:
        yaml.dump(info, f, indent=4)
    print(f"Model saved to {filepath}")
    print(f"Model info saved to {info_filepath}")

def parse_yaml_data(yaml_path):
    """Parse YAML data file in YOLOv5 format"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"Loaded data config from: {yaml_path}")
    return data

def train(args):
    """Train the detection model"""
    print("Loading dataset...")
    # Xác định anchors và num_anchors (có thể thêm vào args nếu muốn tùy chỉnh)
    anchors = ANCHORS
    num_anchors = NUM_ANCHORS

    # Kiểm tra xem data_path có phải là file yaml không
    if args.data_path.endswith('.yaml') or args.data_path.endswith('.yml'):
        with open(args.data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        train_path = os.path.join(os.path.dirname(args.data_path), data_config['train'])
        val_path = os.path.join(os.path.dirname(args.data_path), data_config['val'])
        num_classes = data_config['nc']
        class_names = data_config['names']
        print(f"Loaded dataset config from {args.data_path}")
        print(f"Train path: {train_path}")
        print(f"Validation path: {val_path}")
        print(f"Number of classes: {num_classes}")

        # Tạo dataset từ file paths, truyền anchors và num_anchors
        train_dataset = YOLODatasetFromPaths(
            train_path,
            img_size=args.img_size,
            grid_size=args.grid_size,
            batch_size=args.batch_size,
            num_classes=num_classes,
            class_names=class_names,
            anchors=anchors, # Thêm
            num_anchors=num_anchors, # Thêm
            augment=True
        )
        val_dataset = YOLODatasetFromPaths(
            val_path,
            img_size=args.img_size,
            grid_size=args.grid_size,
            batch_size=args.batch_size,
            num_classes=num_classes,
            class_names=class_names,
            anchors=anchors, # Thêm
            num_anchors=num_anchors, # Thêm
            augment=False
        )
        # Cập nhật lại grid_size nếu dataset tự điều chỉnh
        args.grid_size = train_dataset.grid_size
    else:
        # Sử dụng cách cũ, truyền anchors và num_anchors
        train_dataset, val_dataset, num_classes = get_data_loaders(
            args.data_path,
            img_size=args.img_size,
            grid_size=args.grid_size,
            batch_size=args.batch_size,
            anchors=anchors, # Thêm
            num_anchors=num_anchors # Thêm
        )
        class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else None
        # Cập nhật lại grid_size nếu dataset tự điều chỉnh
        args.grid_size = train_dataset.grid_size

    print(f"Building model with {num_classes} classes, grid size {args.grid_size}, and {num_anchors} anchors...")
    # Truyền anchors và num_anchors vào build_detection_model
    model = build_detection_model(
        input_shape=(args.img_size, args.img_size, 3),
        grid_size=args.grid_size,
        num_classes=num_classes,
        num_anchors=num_anchors, # Thêm
        anchors=anchors,         # Thêm
        learning_rate=args.learning_rate
    )
    model.summary()

    # Callbacks
    log_dir = os.path.join("logs", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, "best_model.keras")

    # Cập nhật ModelCheckpoint để theo dõi val_AP hoặc val_loss
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_AP', # Theo dõi val_AP
        save_best_only=True,
        mode='max',      # Lưu model có val_AP cao nhất
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)

    # Khởi tạo DetectionMetricsCallback với anchors và num_anchors
    metrics_callback = DetectionMetricsCallback(
        validation_data=val_dataset,
        grid_size=args.grid_size,
        num_classes=num_classes,
        anchors=anchors, # Thêm
        num_anchors=num_anchors, # Thêm
        iou_threshold=0.5, # Ngưỡng IoU chuẩn cho mAP@0.5
        confidence_threshold=0.1, # Ngưỡng tin cậy thấp để tính AP
        nms_threshold=0.5, # Ngưỡng NMS khi xử lý dự đoán
        frequency=args.metrics_freq # Tần suất tính metrics
    )

    callbacks = [checkpoint, reduce_lr, early_stopping, tensorboard, metrics_callback]

    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    print("Training finished.")

    # Lưu model cuối cùng kèm thông tin
    final_model_path = os.path.join(log_dir, "final_model.keras")
    model_info = {
        'img_size': args.img_size,
        'grid_size': args.grid_size,
        'num_classes': num_classes,
        'class_names': class_names,
        'anchors': anchors.tolist(), # Lưu anchors vào info
        'num_anchors': num_anchors
    }
    save_model_with_info(model, final_model_path, model_info)

    # Vẽ đồ thị loss và metrics (AP/AR)
    try:
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # AP/AR plot
        # Lấy history từ callback
        ap_history = metrics_callback.ap_history
        ar_history = metrics_callback.ar_history
        epochs_calculated = range(args.metrics_freq -1, args.epochs, args.metrics_freq)
        epochs_calculated = epochs_calculated[:len(ap_history)] # Đảm bảo số epoch khớp

        if ap_history and ar_history:
            plt.subplot(1, 2, 2)
            plt.plot(epochs_calculated, ap_history, label='Validation AP@0.5', marker='o')
            plt.plot(epochs_calculated, ar_history, label='Validation AR@0.5', marker='x')
            plt.title('Validation AP & AR')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.ylim(0, 1)

        plt.tight_layout()
        plot_path = os.path.join(log_dir, "training_plots.png")
        plt.savefig(plot_path)
        print(f"Training plots saved to {plot_path}")
        # plt.show()
    except Exception as e:
        print(f"Error plotting training history: {e}")

def evaluate(model_path, data_path, img_size=64, grid_size=8, batch_size=4):
    """Evaluate the model on a few validation images and plot results."""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False) # Không cần compile lại loss

    # Load model info
    info_path = model_path.replace('.keras', '_info.json')
    if not os.path.exists(info_path):
        print(f"Error: Model info file not found at {info_path}")
        # Fallback hoặc lấy thông tin từ model nếu có thể
        try:
            img_size = model.input_shape[1]
            grid_size = model.output_shape[1]
            num_anchors = model.output_shape[3]
            num_classes = model.output_shape[4] - 5
            anchors = ANCHORS # Sử dụng anchors mặc định nếu không có info
            class_names = None
            print("Warning: Using default anchors and inferred parameters.")
        except Exception as e:
            print(f"Could not infer parameters from model shape. Error: {e}")
            return
    else:
        with open(info_path, 'r') as f:
            model_info = yaml.safe_load(f)
        img_size = model_info['img_size']
        grid_size = model_info['grid_size']
        num_classes = model_info['num_classes']
        class_names = model_info['class_names']
        anchors = np.array(model_info['anchors']) # Load anchors từ info
        num_anchors = model_info['num_anchors']

    print("Loading validation data...")
    # Tạo val_dataset với thông tin đã load
    if data_path.endswith('.yaml') or data_path.endswith('.yml'):
         with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
         val_path = os.path.join(os.path.dirname(data_path), data_config['val'])
         val_dataset = YOLODatasetFromPaths(
            val_path, img_size=img_size, grid_size=grid_size, batch_size=batch_size,
            num_classes=num_classes, class_names=class_names,
            anchors=anchors, num_anchors=num_anchors, augment=False, verbose=False
         )
    else:
        _, val_dataset, _ = get_data_loaders(
            data_path, img_size=img_size, grid_size=grid_size, batch_size=batch_size,
            anchors=anchors, num_anchors=num_anchors
        )

    # Lấy một batch để đánh giá
    try:
        batch_x, batch_y = next(iter(val_dataset))
    except StopIteration:
        print("Validation dataset is empty or too small.")
        return

    print("Running predictions...")
    predictions = model.predict(batch_x)

    print("Processing predictions...")
    # Truyền anchors và num_anchors vào process_predictions
    all_boxes, all_scores, all_classes = process_predictions(
        predictions,
        grid_size=grid_size,
        confidence_threshold=0.3, # Ngưỡng tin cậy để visualize
        nms_threshold=0.5,
        num_classes=num_classes,
        anchors=anchors, # Thêm
        num_anchors=num_anchors # Thêm
    )

    print("Plotting results...")
    # Sử dụng hàm plot_detection_results từ utils
    fig = plot_detection_results(
        images=batch_x, # Ảnh đã chuẩn hóa [0,1]
        all_boxes=all_boxes,
        all_scores=all_scores,
        all_classes=all_classes,
        class_names=class_names,
        figsize=(10, 10)
    )
    eval_plot_path = os.path.join(os.path.dirname(model_path), "evaluation_results.png")
    fig.savefig(eval_plot_path)
    print(f"Evaluation plot saved to {eval_plot_path}")
    # plt.show()

def inference(model_path, image_path, output_path=None):
    """Run inference on a single image."""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load model info
    info_path = model_path.replace('.keras', '_info.json')
    if not os.path.exists(info_path):
        print(f"Error: Model info file not found at {info_path}")
        # Fallback
        try:
            img_size = model.input_shape[1]
            grid_size = model.output_shape[1]
            num_anchors = model.output_shape[3]
            num_classes = model.output_shape[4] - 5
            anchors = ANCHORS
            class_names = None
            print("Warning: Using default anchors and inferred parameters.")
        except Exception as e:
            print(f"Could not infer parameters from model shape. Error: {e}")
            return
    else:
        with open(info_path, 'r') as f:
            model_info = yaml.safe_load(f)
        img_size = model_info['img_size']
        grid_size = model_info['grid_size']
        num_classes = model_info['num_classes']
        class_names = model_info['class_names']
        anchors = np.array(model_info['anchors'])
        num_anchors = model_info['num_anchors']

    print(f"Loading and preprocessing image {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    print("Running prediction...")
    predictions = model.predict(img_batch)

    print("Processing predictions...")
    # Truyền anchors và num_anchors vào process_predictions
    all_boxes, all_scores, all_classes = process_predictions(
        predictions,
        grid_size=grid_size,
        confidence_threshold=0.3, # Ngưỡng tin cậy cho inference
        nms_threshold=0.5,
        num_classes=num_classes,
        anchors=anchors, # Thêm
        num_anchors=num_anchors # Thêm
    )

    # Lấy kết quả cho ảnh đầu tiên (và duy nhất) trong batch
    boxes = all_boxes[0]
    scores = all_scores[0]
    classes = all_classes[0]

    print(f"Found {len(boxes)} objects.")

    # Vẽ bounding boxes lên ảnh gốc
    img_with_boxes = draw_boxes(
        img, # Vẽ lên ảnh gốc chưa resize
        boxes, # Tọa độ đã chuẩn hóa [0,1]
        scores,
        classes.astype(int),
        class_names=class_names
    )

    # Lưu hoặc hiển thị ảnh kết quả
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_with_boxes)
        print(f"Output image saved to {output_path}")
    else:
        # Hiển thị bằng matplotlib (vì cv2.imshow có thể không hoạt động tốt trong mọi môi trường)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate a ResNet-YOLO detection model.')
    subparsers = parser.add_subparsers(dest='mode', help='Select mode: train, evaluate, or inference')

    # --- Train arguments ---
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--data_path', type=str, required=True, help='Path to dataset directory (containing train/val) or YAML file')
    parser_train.add_argument('--img_size', type=int, default=64, help='Input image size')
    parser_train.add_argument('--grid_size', type=int, default=8, help='Grid size for the detection head')
    parser_train.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser_train.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser_train.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser_train.add_argument('--run_name', type=str, default='yolo_run', help='Name for the training run (logs directory)')
    parser_train.add_argument('--metrics_freq', type=int, default=1, help='Frequency (in epochs) to calculate validation AP/AR')

    # --- Evaluate arguments ---
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate the model on validation data')
    parser_eval.add_argument('--model_path', type=str, required=True, help='Path to the trained .keras model file')
    parser_eval.add_argument('--data_path', type=str, required=True, help='Path to dataset directory (containing val) or YAML file')
    # img_size, grid_size sẽ được đọc từ model info

    # --- Inference arguments ---
    parser_infer = subparsers.add_parser('inference', help='Run inference on a single image')
    parser_infer.add_argument('--model_path', type=str, required=True, help='Path to the trained .keras model file')
    parser_infer.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser_infer.add_argument('--output_path', type=str, default=None, help='Path to save the output image with detections (optional)')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        # Không cần truyền img_size, grid_size vì sẽ đọc từ info
        evaluate(args.model_path, args.data_path)
    elif args.mode == 'inference':
        inference(args.model_path, args.image_path, args.output_path)
    else:
        parser.print_help()