import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils import process_predictions, draw_boxes, ANCHORS, NUM_ANCHORS
import json
import yaml

class EasyPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.img_size = None
        self.grid_size = None
        self.num_classes = None
        self.class_names = None
        self.anchors = None
        self.num_anchors = None
        self._load_model_and_info()

    def _load_model_and_info(self):
        """Load the Keras model and its associated info file."""
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Load model info
        info_path = self.model_path.replace('.keras', '_info.json')
        if not os.path.exists(info_path):
            print(f"Warning: Model info file not found at {info_path}")
            # Fallback: Thử suy luận từ model shape và dùng anchor mặc định
            try:
                self.img_size = self.model.input_shape[1]
                self.grid_size = self.model.output_shape[1]
                self.num_anchors = self.model.output_shape[3]
                self.num_classes = self.model.output_shape[4] - 5
                self.anchors = ANCHORS # Sử dụng anchors mặc định
                self.class_names = None
                print(f"Warning: Using default anchors ({self.num_anchors}) and inferred parameters (img_size={self.img_size}, grid_size={self.grid_size}, num_classes={self.num_classes}).")
            except Exception as e:
                print(f"Could not infer parameters from model shape. Error: {e}")
                # Set defaults if inference fails
                self.img_size = 64
                self.grid_size = 8
                self.num_classes = 1
                self.anchors = ANCHORS
                self.num_anchors = NUM_ANCHORS
                self.class_names = None
                print("Falling back to default parameters.")
        else:
            with open(info_path, 'r') as f:
                model_info = yaml.safe_load(f)
            self.img_size = model_info['img_size']
            self.grid_size = model_info['grid_size']
            self.num_classes = model_info['num_classes']
            self.class_names = model_info['class_names']
            # Load anchors từ info, chuyển thành numpy array
            self.anchors = np.array(model_info.get('anchors', ANCHORS)) # Fallback nếu không có trong info
            self.num_anchors = model_info.get('num_anchors', NUM_ANCHORS)
            print(f"Loaded model info: img_size={self.img_size}, grid_size={self.grid_size}, num_classes={self.num_classes}, num_anchors={self.num_anchors}")

    def predict_image(self, image_path, confidence_threshold=0.3, nms_threshold=0.5):
        """Predict objects in a single image."""
        if not self.model:
            print("Model not loaded.")
            return None, None, None, None

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None, None, None, None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        predictions = self.model.predict(img_batch)

        # Truyền anchors và num_anchors vào process_predictions
        all_boxes, all_scores, all_classes = process_predictions(
            predictions,
            grid_size=self.grid_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            num_classes=self.num_classes,
            anchors=self.anchors, # Truyền anchors đã load
            num_anchors=self.num_anchors # Truyền num_anchors đã load
        )

        # Lấy kết quả cho ảnh đầu tiên
        boxes = all_boxes[0]
        scores = all_scores[0]
        classes = all_classes[0]

        return img, boxes, scores, classes

    def draw_and_show(self, image, boxes, scores, classes, output_path=None):
        """Draw bounding boxes on the image and show/save it."""
        img_with_boxes = draw_boxes(
            image, # Ảnh gốc
            boxes, # Tọa độ chuẩn hóa [0,1]
            scores,
            classes.astype(int),
            class_names=self.class_names
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_with_boxes)
            print(f"Output image saved to {output_path}")
        else:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        return img_with_boxes

# Example usage:
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_FILE = "logs/yolo_run/best_model.keras" # Đường dẫn tới model đã train
    IMAGE_TO_PREDICT = "path/to/your/image.jpg" # Đường dẫn tới ảnh cần dự đoán
    OUTPUT_IMAGE_PATH = "predicted_image.jpg" # Đường dẫn lưu ảnh kết quả (hoặc None để hiển thị)
    CONFIDENCE = 0.4 # Ngưỡng tin cậy
    # ---------------------

    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found: {MODEL_FILE}")
        exit()
    if not os.path.exists(IMAGE_TO_PREDICT):
        print(f"Image file not found: {IMAGE_TO_PREDICT}")
        exit()

    # 1. Create predictor instance
    predictor = EasyPredictor(MODEL_FILE)

    # 2. Predict on the image
    original_image, detected_boxes, detected_scores, detected_classes = predictor.predict_image(
        IMAGE_TO_PREDICT,
        confidence_threshold=CONFIDENCE
    )

    # 3. Draw results and show/save
    if original_image is not None:
        print(f"Found {len(detected_boxes)} objects.")
        predictor.draw_and_show(
            original_image,
            detected_boxes,
            detected_scores,
            detected_classes,
            output_path=OUTPUT_IMAGE_PATH
        )