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

class DetectionMetricsCallback(Callback):
    """Callback để tính toán và hiển thị AP và AR sau mỗi epoch trên toàn bộ validation set"""
    def __init__(self, validation_data, grid_size, num_classes, iou_threshold=0.5, confidence_threshold=0.1, nms_threshold=0.5):
        super().__init__()
        self.validation_data = validation_data
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold # Thêm ngưỡng tin cậy cho xử lý dự đoán
        self.nms_threshold = nms_threshold # Thêm ngưỡng NMS
        self.ap_history = []
        self.ar_history = []

    def on_epoch_end(self, epoch, logs=None):
        print("\nCalculating validation metrics...")
        all_true_boxes = []
        all_true_classes = []
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_classes = []

        # Lặp qua toàn bộ validation dataset
        # Sử dụng tqdm để hiển thị thanh tiến trình
        for batch_x, batch_y in tqdm(self.validation_data, desc="Validation Metrics"):
            # Dự đoán trên batch hiện tại
            predictions = self.model.predict(batch_x, verbose=0) # Tắt verbose của predict

            # Xử lý dự đoán cho batch này (áp dụng confidence threshold và NMS)
            batch_pred_boxes, batch_pred_scores, batch_pred_classes = process_predictions(
                predictions,
                grid_size=self.grid_size,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                num_classes=self.num_classes
            )

            # Trích xuất ground truth cho batch này
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

        # Thêm vào lịch sử
        self.ap_history.append(ap)
        self.ar_history.append(ar)

        # Thêm vào logs
        logs = logs or {}
        logs['val_AP'] = ap
        logs['val_AR'] = ar

        # Hiển thị (Keras sẽ tự động hiển thị từ logs)
        # print(f" - val_AP: {ap:.4f} - val_AR: {ar:.4f}") # Không cần in ở đây nữa

    def _extract_true_boxes_batch(self, batch_y):
        """Trích xuất ground truth boxes và classes từ một batch target"""
        batch_true_boxes = []
        batch_true_classes = []
        for y_true in batch_y:
            true_boxes_img = []
            true_classes_img = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if y_true[row, col, 4] > 0:  # Nếu có object
                        x_cell, y_cell, w_cell, h_cell = y_true[row, col, 0:4]
                        x_center = (col + x_cell) / self.grid_size
                        y_center = (row + y_cell) / self.grid_size
                        w = w_cell / self.grid_size
                        h = h_cell / self.grid_size
                        x_min = max(0, x_center - w/2)
                        y_min = max(0, y_center - h/2)
                        x_max = min(1, x_center + w/2)
                        y_max = min(1, y_center + h/2)
                        class_id = np.argmax(y_true[row, col, 5:5+self.num_classes])
                        true_boxes_img.append([x_min, y_min, x_max, y_max])
                        true_classes_img.append(class_id)
            batch_true_boxes.append(np.array(true_boxes_img) if true_boxes_img else np.zeros((0, 4)))
            batch_true_classes.append(np.array(true_classes_img) if true_classes_img else np.array([]))
        return batch_true_boxes, batch_true_classes

    def calculate_mean_ap_ar(self, all_true_boxes, all_true_classes, all_pred_boxes, all_pred_scores, all_pred_classes):
        """Tính toán mAP và mAR trên toàn bộ dataset"""
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

            if num_gt == 0 or len(pred_boxes) == 0:
                # Nếu không có GT hoặc không có dự đoán, thêm FP cho các dự đoán (nếu có)
                for score in pred_scores:
                    all_scores_list.append(score)
                    all_tp_fp_list.append(0) # FP
                continue

            gt_used = np.zeros(num_gt, dtype=bool)

            # Sắp xếp dự đoán theo điểm số giảm dần
            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_classes = pred_classes[order]

            # Tính IoU matrix
            iou_matrix = self.calculate_iou_matrix(pred_boxes, true_boxes)

            # Xác định TP/FP cho từng dự đoán
            for j in range(len(pred_boxes)):
                pred_class = pred_classes[j]
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

        recall = cum_tp / total_num_gt if total_num_gt > 0 else np.zeros_like(cum_tp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9) # Thêm epsilon để tránh chia cho 0

        ap = self.calculate_ap_from_precision_recall(precision, recall)
        ar = total_matched_gt / total_num_gt if total_num_gt > 0 else 0.0

        return ap, ar

    # --- Các hàm calculate_iou_matrix và calculate_ap_from_precision_recall giữ nguyên ---
    def calculate_iou_matrix(self, boxes1, boxes2):
        """Tính ma trận IoU giữa hai tập hợp boxes"""
        # ... (giữ nguyên code gốc) ...
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
        # ... (giữ nguyên code gốc) ...
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
        # Thêm callback để tính AP và AR - THAY ĐỔI TẠI ĐÂY
        DetectionMetricsCallback(
            validation_data=val_dataset,
            grid_size=args.grid_size,
            num_classes=num_classes,
            iou_threshold=0.3,  # Thêm tham số này với giá trị 0.3
            confidence_threshold=0.1, # Ngưỡng tin cậy để xử lý dự đoán
            nms_threshold=0.5        # Ngưỡng NMS
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
    print("Evaluating model...")
    
    # Get a batch of data for visualization
    batch_x, batch_y = next(iter(dataset))
    
    # Make predictions
    predictions = model.predict(batch_x)
    
    # Process predictions
    all_boxes, all_scores, all_classes = process_predictions(
        predictions, 
        grid_size=grid_size,
        confidence_threshold=0.3,
        nms_threshold=0.5,
        num_classes=num_classes
    )
    
    # Visualize results
    class_names = dataset.class_names if hasattr(dataset, 'class_names') else None
    fig = plot_detection_results(
        batch_x, all_boxes, all_scores, all_classes, 
        class_names=class_names
    )
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'detection_results.png'))
    
    # Show individual images with detections
    for i in range(min(5, len(batch_x))):
        image = batch_x[i]
        boxes = all_boxes[i]
        scores = all_scores[i]
        class_ids = all_classes[i]
        
        # Draw bounding boxes on the image
        result_img = draw_boxes(
            image, boxes, scores, class_ids, 
            class_names=class_names
        )
        
        # Convert back to RGB for display
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        if output_dir:
            cv2.imwrite(
                os.path.join(output_dir, f'detection_sample_{i}.jpg'),
                result_img * 255
            )

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
    
    # Evaluate if requested
    if args.evaluate:
        import cv2  # Import here as it's used in evaluate
        evaluate(model, val_dataset, num_classes, args.grid_size, args.output_dir)