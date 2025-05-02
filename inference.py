import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from utils import process_predictions, draw_boxes

def load_model(model_path):
    """Load the trained detection model"""
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(
            model_path, 
            compile=False  # Don't need the custom loss function for inference
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_class_names(class_file):
    """Load class names from a file"""
    if not os.path.exists(class_file):
        print(f"Class file {class_file} not found. Using class indices instead.")
        return None
    
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(class_names)} class names")
    return class_names

def run_inference(model, image_path, class_names=None, img_size=64, grid_size=8, 
                  confidence_threshold=0.3, output_dir=None):
    """Run inference on an image and visualize results"""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found")
        return
    
    # Determine number of classes from model's output shape
    num_classes = model.output_shape[-1] - 5  # 5 for x, y, w, h, objectness
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original image dimensions
    orig_h, orig_w = img.shape[:2]
    
    # Resize for the model
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
        confidence_threshold=confidence_threshold,
        nms_threshold=0.5,
        num_classes=num_classes
    )
    
    # Get boxes, scores and classes for the first (and only) image
    boxes = all_boxes[0]
    scores = all_scores[0]
    class_ids = all_classes[0]
    
    # Draw boxes on the normalized image
    result_img_norm = draw_boxes(
        img_norm, boxes, scores, class_ids, 
        class_names=class_names
    )
    
    # Convert back to original image size and RGB for display
    result_img = cv2.cvtColor(result_img_norm, cv2.COLOR_RGB2BGR)
    result_img = cv2.resize(result_img, (orig_w, orig_h))
    
    # Save or display the result
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, 
            os.path.basename(image_path).replace('.', '_detected.')
        )
        cv2.imwrite(output_path, result_img * 255)
        print(f"Saved detection result to {output_path}")
    
    # Print detection results
    print(f"Found {len(boxes)} objects in the image:")
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        class_name = class_names[int(class_id)] if class_names else f"Class {int(class_id)}"
        x_min, y_min, x_max, y_max = box
        print(f"  {i+1}. {class_name}: {score:.2f} at [{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}]")
    
    return result_img_norm, boxes, scores, class_ids

def batch_inference(model, image_dir, class_names=None, img_size=64, grid_size=8, 
                    confidence_threshold=0.3, output_dir=None):
    """Run inference on all images in a directory"""
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} not found")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(os.path.join(image_dir, filename))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Process each image
    results = []
    for image_path in image_files:
        print(f"Processing {os.path.basename(image_path)}...")
        result = run_inference(
            model, 
            image_path, 
            class_names=class_names, 
            img_size=img_size, 
            grid_size=grid_size,
            confidence_threshold=confidence_threshold,
            output_dir=output_dir
        )
        results.append(result)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained detection model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model (.h5 file)')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to a single image for inference')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Path to directory of images for batch inference')
    parser.add_argument('--class_file', type=str, default=None,
                       help='Path to the class names file')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save inference results')
    parser.add_argument('--img_size', type=int, default=64,
                       help='Image size (square) for the model')
    parser.add_argument('--grid_size', type=int, default=8,
                       help='Detection grid size')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Check if either image_path or image_dir is provided
    if args.image_path is None and args.image_dir is None:
        parser.error("Either --image_path or --image_dir must be provided")
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        exit(1)
    
    # Load class names if provided
    class_names = load_class_names(args.class_file) if args.class_file else None
    
    # Run inference
    if args.image_path:
        # Single image inference
        run_inference(
            model, 
            args.image_path, 
            class_names=class_names, 
            img_size=args.img_size,
            grid_size=args.grid_size,
            confidence_threshold=args.confidence,
            output_dir=args.output_dir
        )
    elif args.image_dir:
        # Batch inference
        batch_inference(
            model, 
            args.image_dir, 
            class_names=class_names, 
            img_size=args.img_size,
            grid_size=args.grid_size,
            confidence_threshold=args.confidence,
            output_dir=args.output_dir
        ) 