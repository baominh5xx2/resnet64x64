import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback

from dataset import get_data_loaders, YOLODatasetFromPaths
from detection_model import build_detection_model
from utils import process_predictions, draw_boxes, plot_detection_results

class DetectionMetricsCallback(Callback):
    """Callback để tính toán và hiển thị AP và AR sau mỗi epoch"""
    def __init__(self, validation_data, grid_size, num_classes, iou_threshold=0.5):
        super().__init__()
        self.validation_data = validation_data
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.ap_history = []
        self.ar_history = []
        
        # Tạo một validation subset chỉ chứa ảnh có label
        self.valid_images_batch_x = []
        self.valid_images_batch_y = []
        self.has_valid_images = False
        self._prepare_valid_images_subset()
        
    def _prepare_valid_images_subset(self):
        """Lọc validation set để chỉ giữ những ảnh có label"""
        print("Lọc validation set để tìm ảnh có label...")
        
        # Lặp qua validation set để tìm ảnh có label
        for batch_x, batch_y in self.validation_data:
            # Kiểm tra từng ảnh trong batch
            for i in range(len(batch_x)):
                # Kiểm tra xem ảnh có chứa ít nhất một đối tượng không 
                # (objectness > 0 ở bất kỳ grid cell nào)
                if np.sum(batch_y[i, ..., 4]) > 0:
                    self.valid_images_batch_x.append(np.expand_dims(batch_x[i], axis=0))
                    self.valid_images_batch_y.append(np.expand_dims(batch_y[i], axis=0))
            
            # Chỉ quét một số batch để tránh quá lâu
            if len(self.valid_images_batch_x) >= 20:
                break
        
        if len(self.valid_images_batch_x) > 0:
            print(f"Đã tìm thấy {len(self.valid_images_batch_x)} ảnh có label trong validation set")
            self.has_valid_images = True
        else:
            print("CẢNH BÁO: Không tìm thấy ảnh có label trong validation set!")
        
    def on_epoch_end(self, epoch, logs=None):
        if not self.has_valid_images:
            print(" - val_AP: 0.0000 - val_AR: 0.0000 (không có ảnh có label)")
            return
        
        # Kết hợp các ảnh có label thành một batch để đánh giá
        batch_x = np.vstack(self.valid_images_batch_x)
        batch_y = np.vstack(self.valid_images_batch_y)
        
        # Dự đoán
        predictions = self.model.predict(batch_x)
        
        # Tính AP và AR
        ap, ar = self.calculate_metrics(batch_y, predictions)
        
        # Thêm vào lịch sử
        self.ap_history.append(ap)
        self.ar_history.append(ar)
        
        # Thêm vào logs
        logs = logs or {}
        logs['val_AP'] = ap
        logs['val_AR'] = ar
        
        # Hiển thị
        print(f" - val_AP: {ap:.4f} - val_AR: {ar:.4f}")
    
    def calculate_metrics(self, true_y, pred_y):
        """Tính toán AP và AR từ ground truth và dự đoán"""
        batch_size = len(true_y)
        total_ap = 0.0
        total_ar = 0.0
        
        for b in range(batch_size):
            # Lấy ground truth
            y_true = true_y[b]
            
            # Lấy dự đoán
            y_pred = pred_y[b]
            
            # Tạo danh sách boxes từ ground truth
            true_boxes = []
            true_classes = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if y_true[row, col, 4] > 0:  # Nếu có object
                        # Lấy tọa độ và kích thước từ ground truth
                        x_cell, y_cell, w_cell, h_cell = y_true[row, col, 0:4]
                        
                        # Chuyển sang tọa độ hình ảnh
                        x_center = (col + x_cell) / self.grid_size
                        y_center = (row + y_cell) / self.grid_size
                        w = w_cell / self.grid_size
                        h = h_cell / self.grid_size
                        
                        # Chuyển sang format [x_min, y_min, x_max, y_max]
                        x_min = max(0, x_center - w/2)
                        y_min = max(0, y_center - h/2)
                        x_max = min(1, x_center + w/2)
                        y_max = min(1, y_center + h/2)
                        
                        # Lớp
                        class_id = np.argmax(y_true[row, col, 5:5+self.num_classes])
                        
                        true_boxes.append([x_min, y_min, x_max, y_max])
                        true_classes.append(class_id)
            
            # Xử lý dự đoán
            pred_boxes = []
            pred_scores = []
            pred_classes = []
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    # Lấy dự đoán cho ô lưới này
                    cell_pred = y_pred[row, col]
                    
                    # Lấy tọa độ và kích thước
                    x_cell, y_cell, w_cell, h_cell = cell_pred[0:4]
                    objectness = cell_pred[4]
                    
                    # Bỏ qua nếu objectness thấp
                    if objectness < 0.00005:
                        continue
                    
                    # Lấy class
                    class_probs = cell_pred[5:5+self.num_classes]
                    class_id = np.argmax(class_probs)
                    class_score = class_probs[class_id]
                    
                    # Điểm số cuối cùng (objectness * class probability)
                    score = objectness * class_score
                    
                    # Bỏ qua nếu điểm số thấp
                    if score < 0.00005:
                        continue
                    
                    # Chuyển sang tọa độ hình ảnh
                    x_center = (col + x_cell) / self.grid_size
                    y_center = (row + y_cell) / self.grid_size
                    w = w_cell / self.grid_size
                    h = h_cell / self.grid_size
                    
                    # Chuyển sang format [x_min, y_min, x_max, y_max]
                    x_min = max(0, x_center - w/2)
                    y_min = max(0, y_center - h/2)
                    x_max = min(1, x_center + w/2)
                    y_max = min(1, y_center + h/2)
                    
                    pred_boxes.append([x_min, y_min, x_max, y_max])
                    pred_scores.append(score)
                    pred_classes.append(class_id)
            
            # Chuyển sang numpy arrays
            if len(true_boxes) > 0:
                true_boxes = np.array(true_boxes)
                true_classes = np.array(true_classes)
            else:
                true_boxes = np.zeros((0, 4))
                true_classes = np.array([])
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array(pred_boxes)
                pred_scores = np.array(pred_scores)
                pred_classes = np.array(pred_classes)
            else:
                pred_boxes = np.zeros((0, 4))
                pred_scores = np.array([])
                pred_classes = np.array([])
            
            # Tính AP và AR cho hình ảnh này
            ap, ar = self.calculate_ap_ar(true_boxes, true_classes, pred_boxes, pred_scores, pred_classes)
            
            total_ap += ap
            total_ar += ar
        
        # Trung bình trên batch
        mean_ap = total_ap / batch_size if batch_size > 0 else 0
        mean_ar = total_ar / batch_size if batch_size > 0 else 0
        
        return mean_ap, mean_ar
    
    def calculate_ap_ar(self, true_boxes, true_classes, pred_boxes, pred_scores, pred_classes):
        """Tính toán AP và AR cho một hình ảnh"""
        # Nếu không có ground truth hoặc dự đoán
        if len(true_boxes) == 0 or len(pred_boxes) == 0:
            return 0.0, 0.0
        
        # Số lượng ground truth
        num_gt = len(true_boxes)
        
        # Khởi tạo mảng đã sử dụng để đánh dấu ground truth đã match
        gt_used = np.zeros(num_gt, dtype=bool)
        
        # Sắp xếp dự đoán theo điểm số giảm dần
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_classes = pred_classes[order]
        
        # Tính IoU cho mỗi cặp (predicted, ground truth)
        iou_matrix = self.calculate_iou_matrix(pred_boxes, true_boxes)
        
        # Đếm số TP và FP
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        
        for i in range(len(pred_boxes)):
            # Chỉ xét các ground truth có cùng lớp
            gt_same_class = np.where(true_classes == pred_classes[i])[0]
            
            # Nếu không có ground truth cùng lớp, là FP
            if len(gt_same_class) == 0:
                fp[i] = 1
                continue
            
            # Lấy IoU với các ground truth cùng lớp
            ious = iou_matrix[i, gt_same_class]
            
            # Lấy GT có IoU cao nhất
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            gt_idx = gt_same_class[max_iou_idx]
            
            # Nếu IoU > threshold và GT chưa dùng, đây là TP
            if max_iou >= self.iou_threshold and not gt_used[gt_idx]:
                tp[i] = 1
                gt_used[gt_idx] = True
            else:
                fp[i] = 1
        
        # Tính precision và recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / num_gt if num_gt > 0 else np.zeros_like(cum_tp)
        precision = cum_tp / (cum_tp + cum_fp)
        
        # Tính AP bằng phương pháp AUC
        ap = self.calculate_ap_from_precision_recall(precision, recall)
        
        # Tính AR (trung bình recall ở các threshold IoU)
        ar = np.sum(gt_used) / num_gt if num_gt > 0 else 0.0
        
        return ap, ar
    
    def calculate_iou_matrix(self, boxes1, boxes2):
        """Tính ma trận IoU giữa hai tập hợp boxes"""
        # Xử lý trường hợp mảng rỗng
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        # Khởi tạo ma trận IoU
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        # Tính IoU cho từng cặp boxes
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                # Tính phần giao
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                # Diện tích phần giao
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                
                # Diện tích của từng box
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                
                # Diện tích phần hợp
                union_area = box1_area + box2_area - inter_area
                
                # IoU
                iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0
        
        return iou_matrix
    
    def calculate_ap_from_precision_recall(self, precision, recall):
        """Tính AP từ precision và recall bằng phương pháp AUC"""
        # Thêm điểm đầu và cuối
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        
        # Làm mịn precision (đảm bảo precision không giảm)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
        
        # Tìm các điểm mà recall thay đổi
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Tính diện tích dưới đường cong precision-recall
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
        
        return ap

def save_model_with_info(model, save_path, grid_size, num_classes):
    """Save model along with important info for prediction"""
    # Đảm bảo đường dẫn kết thúc bằng .keras
    if not save_path.endswith('.keras'):
        save_path = os.path.splitext(save_path)[0] + '.keras'
    
    # Save model weights
    model.save(save_path, save_format='keras')
    
    # Save additional info
    info_path = os.path.splitext(save_path)[0] + '_info.json'
    model_info = {
        'grid_size': grid_size,
        'num_classes': num_classes,
        'input_shape': model.input_shape[1:],
        'output_shape': model.output_shape[1:]
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f)
    
    print(f"Model saved to {save_path}")
    print(f"Model info saved to {info_path}")

def parse_yaml_data(yaml_path):
    """Parse YAML data file in YOLOv5 format"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"Loaded data config from: {yaml_path}")
    return data

def train(args):
    """Train the detection model"""
    print("Loading dataset...")
    
    # Kiểm tra xem data_path có phải là file yaml không
    if args.data_path.endswith('.yaml') or args.data_path.endswith('.yml'):
        # Đọc file yaml
        data_config = parse_yaml_data(args.data_path)
        
        # Lấy đường dẫn đến train.txt và val.txt
        train_path = data_config.get('train', '')
        val_path = data_config.get('val', '')
        
        # Lấy số lớp và tên lớp
        num_classes = data_config.get('nc', 0)
        class_names = data_config.get('names', [f'class_{i}' for i in range(num_classes)])
        
        print(f"Dataset config from YAML:")
        print(f"  - Train path: {train_path}")
        print(f"  - Val path: {val_path}")
        print(f"  - Number of classes: {num_classes}")
        
        # Tạo dataset từ file paths
        train_dataset = YOLODatasetFromPaths(
            train_path,
            img_size=args.img_size,
            grid_size=args.grid_size,
            batch_size=args.batch_size,
            num_classes=num_classes,
            class_names=class_names,
            augment=True
        )
        
        val_dataset = YOLODatasetFromPaths(
            val_path,
            img_size=args.img_size,
            grid_size=args.grid_size,
            batch_size=args.batch_size,
            num_classes=num_classes,
            class_names=class_names,
            augment=False
        )
    else:
        # Sử dụng cách cũ nếu là đường dẫn thư mục
        train_dataset, val_dataset, num_classes = get_data_loaders(
            args.data_path,
            img_size=args.img_size,
            grid_size=args.grid_size,
            batch_size=args.batch_size
        )
        class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else None
    
    print(f"Building model with {num_classes} classes...")
    model = build_detection_model(
        input_shape=(args.img_size, args.img_size, 3),
        grid_size=args.grid_size,
        num_classes=num_classes
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(args.output_dir, 'model_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(args.output_dir, 'logs'),
            write_graph=True,
            update_freq='epoch'
        ),
        # Thêm callback để tính AP và AR
        DetectionMetricsCallback(
            validation_data=val_dataset,
            grid_size=args.grid_size,
            num_classes=num_classes
        )
    ]
    
    # Display model summary
    model.summary()
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1  # Chỉ hiển thị một thanh tiến trình cho mỗi epoch
    )
    
    # Save final model with info
    final_model_path = os.path.join(args.output_dir, 'model_final.keras')
    save_model_with_info(model, final_model_path, args.grid_size, num_classes)
    
    # Plot training history
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Subplot 2: AP and AR
    plt.subplot(1, 2, 2)
    ap_history = [cb.ap_history for cb in callbacks if isinstance(cb, DetectionMetricsCallback)]
    ar_history = [cb.ar_history for cb in callbacks if isinstance(cb, DetectionMetricsCallback)]

    if ap_history and ar_history:
        plt.plot(ap_history[0], label='AP')
        plt.plot(ar_history[0], label='AR')
        plt.title('Precision & Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    return model, train_dataset, val_dataset, num_classes

def evaluate(model, dataset, num_classes, grid_size=8, output_dir=None):
    """Evaluate the model and visualize some predictions"""
    print("Evaluating model on complete validation set...")
    
    # Khởi tạo biến tích lũy cho toàn bộ validation set
    all_true_boxes = []
    all_true_classes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []
    
    # Lặp qua toàn bộ validation set
    num_batches = 0
    print("Collecting predictions...")
    for batch_x, batch_y in dataset:
        # Dự đoán
        predictions = model.predict(batch_x, verbose=0)
        
        # Xử lý từng batch
        for b in range(len(batch_x)):
            # Lấy ground truth
            y_true = batch_y[b]
            
            # Lấy dự đoán
            y_pred = predictions[b]
            
            # Tạo danh sách boxes từ ground truth
            true_boxes = []
            true_classes = []
            for row in range(grid_size):
                for col in range(grid_size):
                    if y_true[row, col, 4] > 0:  # Nếu có object
                        # Lấy tọa độ và kích thước từ ground truth
                        x_cell, y_cell, w_cell, h_cell = y_true[row, col, 0:4]
                        
                        # Chuyển sang tọa độ hình ảnh
                        x_center = (col + x_cell) / grid_size
                        y_center = (row + y_cell) / grid_size
                        w = w_cell / grid_size
                        h = h_cell / grid_size
                        
                        # Chuyển sang format [x_min, y_min, x_max, y_max]
                        x_min = max(0, x_center - w/2)
                        y_min = max(0, y_center - h/2)
                        x_max = min(1, x_center + w/2)
                        y_max = min(1, y_center + h/2)
                        
                        # Lớp
                        class_id = np.argmax(y_true[row, col, 5:5+num_classes])
                        
                        true_boxes.append([x_min, y_min, x_max, y_max])
                        true_classes.append(class_id)
            
            # Xử lý dự đoán
            pred_boxes = []
            pred_scores = []
            pred_classes = []
            
            for row in range(grid_size):
                for col in range(grid_size):
                    # Lấy dự đoán cho ô lưới này
                    cell_pred = y_pred[row, col]
                    
                    # Lấy tọa độ và kích thước
                    x_cell, y_cell, w_cell, h_cell = cell_pred[0:4]
                    objectness = cell_pred[4]
                    
                    # Bỏ qua nếu objectness thấp
                    if objectness < 0.01:  # Ngưỡng cao hơn để lọc nhiễu
                        continue
                    
                    # Lấy class
                    class_probs = cell_pred[5:5+num_classes]
                    class_id = np.argmax(class_probs)
                    class_score = class_probs[class_id]
                    
                    # Điểm số cuối cùng (objectness * class probability)
                    score = objectness * class_score
                    
                    # Bỏ qua nếu điểm số thấp
                    if score < 0.01:  # Ngưỡng cao hơn để lọc nhiễu
                        continue
                    
                    # Chuyển sang tọa độ hình ảnh
                    x_center = (col + x_cell) / grid_size
                    y_center = (row + y_cell) / grid_size
                    w = w_cell / grid_size
                    h = h_cell / grid_size
                    
                    # Chuyển sang format [x_min, y_min, x_max, y_max]
                    x_min = max(0, x_center - w/2)
                    y_min = max(0, y_center - h/2)
                    x_max = min(1, x_center + w/2)
                    y_max = min(1, y_center + h/2)
                    
                    pred_boxes.append([x_min, y_min, x_max, y_max])
                    pred_scores.append(score)
                    pred_classes.append(class_id)
            
            # Thêm vào danh sách tổng
            if true_boxes:
                all_true_boxes.extend(true_boxes)
                all_true_classes.extend(true_classes)
            
            if pred_boxes:
                all_pred_boxes.extend(pred_boxes)
                all_pred_scores.extend(pred_scores)
                all_pred_classes.extend(pred_classes)
        
        num_batches += 1
        if num_batches % 10 == 0:
            print(f"Processed {num_batches} batches...")
    
    print(f"Collected {len(all_true_boxes)} ground truth objects and {len(all_pred_boxes)} predictions")
    
    # Chuyển sang numpy arrays
    all_true_boxes = np.array(all_true_boxes) if all_true_boxes else np.zeros((0, 4))
    all_true_classes = np.array(all_true_classes) if all_true_classes else np.array([])
    all_pred_boxes = np.array(all_pred_boxes) if all_pred_boxes else np.zeros((0, 4))
    all_pred_scores = np.array(all_pred_scores) if all_pred_scores else np.array([])
    all_pred_classes = np.array(all_pred_classes) if all_pred_classes else np.array([])
    
    # Tính toán AP và AR tổng thể
    iou_threshold = 0.5
    print("\n=== EVALUATION RESULTS ===")
    print(f"IoU threshold: {iou_threshold}")
    
    # Tính AP và AR tổng thể
    metrics_calc = DetectionMetricsCallback(None, grid_size, num_classes, iou_threshold)
    overall_ap, overall_ar = metrics_calc.calculate_ap_ar(
        all_true_boxes, all_true_classes, 
        all_pred_boxes, all_pred_scores, all_pred_classes
    )
    
    print(f"\nOverall metrics:")
    print(f"  - Average Precision (AP): {overall_ap:.4f}")
    print(f"  - Average Recall (AR): {overall_ar:.4f}")
    
    # Tính AP và AR cho từng class
    class_names = dataset.class_names if hasattr(dataset, 'class_names') else [f"Class {i}" for i in range(num_classes)]
    
    print("\nPer-class metrics:")
    for c in range(num_classes):
        # Lọc chỉ lấy ground truth của class này
        class_true_indices = np.where(all_true_classes == c)[0]
        class_true_boxes = all_true_boxes[class_true_indices] if len(class_true_indices) > 0 else np.zeros((0, 4))
        class_true_classes = all_true_classes[class_true_indices] if len(class_true_indices) > 0 else np.array([])
        
        # Lọc chỉ lấy predictions của class này
        class_pred_indices = np.where(all_pred_classes == c)[0]
        class_pred_boxes = all_pred_boxes[class_pred_indices] if len(class_pred_indices) > 0 else np.zeros((0, 4))
        class_pred_scores = all_pred_scores[class_pred_indices] if len(class_pred_indices) > 0 else np.array([])
        class_pred_classes = all_pred_classes[class_pred_indices] if len(class_pred_indices) > 0 else np.array([])
        
        # Tính AP và AR cho class này
        class_ap, class_ar = metrics_calc.calculate_ap_ar(
            class_true_boxes, class_true_classes, 
            class_pred_boxes, class_pred_scores, class_pred_classes
        )
        
        print(f"  - {class_names[c]}: AP={class_ap:.4f}, AR={class_ar:.4f}, GT={len(class_true_boxes)}, Pred={len(class_pred_boxes)}")
    
    # Hiển thị một số ảnh ví dụ với detection
    print("\nGenerating visualization examples...")
    batch_x, batch_y = next(iter(dataset))
    predictions = model.predict(batch_x)
    
    # Process predictions for visualization
    all_boxes, all_scores, all_classes = process_predictions(
        predictions, 
        grid_size=grid_size,
        confidence_threshold=0.3,
        nms_threshold=0.5,
        num_classes=num_classes
    )
    
    fig = plot_detection_results(
        batch_x, all_boxes, all_scores, all_classes, 
        class_names=class_names
    )
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'detection_results.png'))
        print(f"Visualization saved to {os.path.join(output_dir, 'detection_results.png')}")

def inference(model, image_path, grid_size=8, num_classes=1, class_names=None, img_size=64):
    """Run inference on a single image"""
    from utils import process_predictions, draw_boxes
    import cv2
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_data = np.expand_dims(img_norm, axis=0)
    
    # Run prediction
    predictions = model.predict(input_data)
    
    # Process predictions
    all_boxes, all_scores, all_classes = process_predictions(
        predictions, 
        grid_size=grid_size,
        confidence_threshold=0.3,
        nms_threshold=0.5,
        num_classes=num_classes
    )
    
    # Draw boxes on image
    result_img = draw_boxes(
        img_norm, all_boxes[0], all_scores[0], all_classes[0], 
        class_names=class_names
    )
    
    return result_img, all_boxes[0], all_scores[0], all_classes[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--data_path', '--data', type=str, required=True, 
                      help='Path to dataset directory or YAML file (required)')
    parser.add_argument('--output_dir', '--name', type=str, default='./output',
                      help='Directory to save models and results')
    parser.add_argument('--img_size', '--img', type=int, default=64,
                      help='Image size (square)')
    parser.add_argument('--grid_size', type=int, default=8,
                      help='Detection grid size')
    parser.add_argument('--batch_size', '--batch', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker processes for data loading')
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate model after training')
    parser.add_argument('--weights', type=str, default=None, 
                      help='[Not implemented] Path to pre-trained weights')
    
    args = parser.parse_args()
    
    # Hiển thị thông số
    print("=== Training Configuration ===")
    print(f"Dataset path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Grid size: {args.grid_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Workers: {args.workers}")
    print(f"Evaluate after training: {args.evaluate}")
    print(f"Pre-trained weights: {args.weights}")
    print("=============================")
    
    if args.weights:
        print("WARNING: Loading pre-trained weights is not implemented yet. Training from scratch.")
    
    # Train the model
    model, train_dataset, val_dataset, num_classes = train(args)
    
    # Evaluate on full validation set
    print("\nEvaluating model on full validation set...")
    evaluate(model, val_dataset, num_classes, args.grid_size, args.output_dir)