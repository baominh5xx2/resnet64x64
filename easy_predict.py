import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils import process_predictions, draw_boxes
import json

def predict_image(model_path, image_path, class_file=None, output_path=None, confidence=0.3):
    """
    Dự đoán đối tượng trong ảnh với bất kỳ kích thước nào và hiển thị/lưu kết quả
    
    Args:
        model_path: Đường dẫn đến model đã train
        image_path: Đường dẫn đến ảnh cần dự đoán
        class_file: Đường dẫn đến file chứa tên các class
        output_path: Đường dẫn để lưu ảnh kết quả
        confidence: Ngưỡng tin cậy cho các dự đoán
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh tại {image_path}")
        return
    
    # Tải model
    print(f"Đang tải model từ {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return
    
    # Kiểm tra xem có file info đi kèm không
    info_path = os.path.splitext(model_path)[0] + '_info.json'
    model_info = None
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print(f"Đã tải thông tin model từ {info_path}")
        except Exception as e:
            print(f"Lỗi khi đọc file info: {e}")
    
    # Xác định kích thước input của model
    input_shape = model.input_shape[1:3]  # [height, width]
    img_size = input_shape[0]  # Giả sử ảnh vuông
    
    # Xác định grid_size từ file info hoặc output shape
    if model_info and 'grid_size' in model_info:
        grid_size = model_info['grid_size']
        print(f"Grid size từ file info: {grid_size}x{grid_size}")
    else:
        # Xác định grid_size từ output shape của model
        output_shape = model.output_shape[1:3]  # [grid_height, grid_width]
        grid_size = output_shape[0]  # Giả sử grid vuông
        print(f"Grid size phát hiện từ model: {grid_size}x{grid_size}")
    
    print(f"Model yêu cầu ảnh kích thước {img_size}x{img_size}")
    
    # Tải tên lớp nếu có
    class_names = None
    if class_file and os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Đã tải {len(class_names)} tên lớp từ {class_file}")
    
    # Tính số lớp từ model hoặc file info
    if model_info and 'num_classes' in model_info:
        num_classes = model_info['num_classes']
        print(f"Số lớp từ file info: {num_classes}")
    else:
        num_classes = model.output_shape[-1] - 5  # 5 là x, y, w, h, objectness
        print(f"Số lớp phát hiện từ model: {num_classes}")
    
    # Đọc ảnh
    print(f"Đọc ảnh từ {image_path}...")
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    original_h, original_w = original_img.shape[:2]
    print(f"Kích thước ảnh gốc: {original_w}x{original_h}")
    
    # Chuyển sang RGB và resize
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Thêm batch dimension
    input_tensor = np.expand_dims(img_norm, axis=0)
    
    # Chạy dự đoán
    print("Đang thực hiện dự đoán...")
    predictions = model.predict(input_tensor)
    
    # Xử lý dự đoán
    all_boxes, all_scores, all_classes = process_predictions(
        predictions, 
        grid_size=grid_size,
        confidence_threshold=confidence,
        nms_threshold=0.5,
        num_classes=num_classes
    )
    
    boxes = all_boxes[0]
    scores = all_scores[0]
    classes = all_classes[0]
    
    # Vẽ kết quả lên ảnh resized
    result_img = draw_boxes(
        img_norm, boxes, scores, classes, 
        class_names=class_names
    )
    
    # Chuyển lại kích thước gốc
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    result_img_original_size = cv2.resize(result_img_bgr, (original_w, original_h))
    
    # Hiển thị kết quả
    print(f"Đã tìm thấy {len(boxes)} đối tượng:")
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        class_name = class_names[int(class_id)] if class_names else f"Lớp {int(class_id)}"
        x_min, y_min, x_max, y_max = box
        print(f"  {i+1}. {class_name}: {score:.2f} tại [{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}]")
    
    # Lưu kết quả nếu yêu cầu
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Lưu kết quả vào {output_path}")
        cv2.imwrite(output_path, result_img_original_size * 255)
    
    # Hiển thị kết quả bằng matplotlib
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Ảnh gốc")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img_original_size, cv2.COLOR_BGR2RGB))
    plt.title("Kết quả phát hiện")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return boxes, scores, classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phát hiện đối tượng trong ảnh với bất kỳ kích thước nào")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến model (.h5 hoặc .keras)")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh cần phân tích")
    parser.add_argument("--classes", type=str, default=None, help="Đường dẫn đến file tên lớp")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn để lưu ảnh kết quả")
    parser.add_argument("--confidence", type=float, default=0.3, help="Ngưỡng tin cậy (0-1)")
    
    args = parser.parse_args()
    
    predict_image(
        model_path=args.model,
        image_path=args.image,
        class_file=args.classes,
        output_path=args.output,
        confidence=args.confidence
    ) 