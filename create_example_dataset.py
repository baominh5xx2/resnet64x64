import os
import random
import shutil
import argparse
import numpy as np
import cv2

def create_directory_structure(base_dir):
    """Create the YOLO-format directory structure"""
    # Create main directories
    os.makedirs(base_dir, exist_ok=True)
    
    # Create train and val directories with their subdirectories
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(base_dir, split, subdir), exist_ok=True)
    
    return os.path.join(base_dir, 'train'), os.path.join(base_dir, 'val')

def generate_random_box(img_size, min_size=0.1, max_size=0.5):
    """Generate a random bounding box in normalized coordinates"""
    # Generate width and height
    width = random.uniform(min_size, max_size)
    height = random.uniform(min_size, max_size)
    
    # Generate center coordinates
    x_center = random.uniform(width/2, 1 - width/2)
    y_center = random.uniform(height/2, 1 - height/2)
    
    return x_center, y_center, width, height

def generate_synthetic_image(img_size, num_shapes=3, class_names=None):
    """Generate a synthetic image with random shapes"""
    # Create a blank image with random background color
    bg_color = (
        random.randint(0, 128),
        random.randint(0, 128),
        random.randint(0, 128)
    )
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * bg_color
    
    # List to store bounding box annotations
    annotations = []
    
    # Number of shapes to generate
    num_shapes = random.randint(1, num_shapes)
    
    for _ in range(num_shapes):
        # Randomly select class
        class_id = random.randint(0, len(class_names)-1 if class_names else 0)
        
        # Generate random box
        x_center, y_center, width, height = generate_random_box(img_size)
        
        # Calculate box bounds in pixel coordinates
        x_min = int((x_center - width/2) * img_size)
        y_min = int((y_center - height/2) * img_size)
        x_max = int((x_center + width/2) * img_size)
        y_max = int((y_center + height/2) * img_size)
        
        # Random shape color
        color = (
            random.randint(128, 255),
            random.randint(128, 255),
            random.randint(128, 255)
        )
        
        # Draw shape based on class
        shape_type = class_id % 3  # 0: rectangle, 1: circle, 2: triangle
        
        if shape_type == 0:  # Rectangle
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, -1)
        elif shape_type == 1:  # Circle
            center = (int(x_center * img_size), int(y_center * img_size))
            radius = int(min(width, height) * img_size / 2)
            cv2.circle(img, center, radius, color, -1)
        else:  # Triangle
            pts = np.array([
                [int(x_center * img_size), y_min],
                [x_min, y_max],
                [x_max, y_max]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
        
        # Add annotation
        annotations.append((class_id, x_center, y_center, width, height))
    
    return img, annotations

def create_dataset(base_dir, num_train=100, num_val=20, img_size=64, num_classes=3):
    """Create a synthetic dataset for object detection training"""
    train_dir, val_dir = create_directory_structure(base_dir)
    
    # Create class names
    class_names = [f"shape_{i}" for i in range(num_classes)]
    
    # Write class names to files
    for split_dir in [train_dir, val_dir]:
        with open(os.path.join(split_dir, 'classes.txt'), 'w') as f:
            f.write('\n'.join(class_names))
    
    # Generate training images and labels
    for i in range(num_train):
        # Generate synthetic image
        img, annotations = generate_synthetic_image(img_size, num_shapes=3, class_names=class_names)
        
        # Save image
        img_path = os.path.join(train_dir, 'images', f'train_{i:04d}.jpg')
        cv2.imwrite(img_path, img)
        
        # Save annotations
        label_path = os.path.join(train_dir, 'labels', f'train_{i:04d}.txt')
        with open(label_path, 'w') as f:
            for anno in annotations:
                f.write(f"{anno[0]} {anno[1]:.6f} {anno[2]:.6f} {anno[3]:.6f} {anno[4]:.6f}\n")
    
    # Generate validation images and labels
    for i in range(num_val):
        # Generate synthetic image
        img, annotations = generate_synthetic_image(img_size, num_shapes=3, class_names=class_names)
        
        # Save image
        img_path = os.path.join(val_dir, 'images', f'val_{i:04d}.jpg')
        cv2.imwrite(img_path, img)
        
        # Save annotations
        label_path = os.path.join(val_dir, 'labels', f'val_{i:04d}.txt')
        with open(label_path, 'w') as f:
            for anno in annotations:
                f.write(f"{anno[0]} {anno[1]:.6f} {anno[2]:.6f} {anno[3]:.6f} {anno[4]:.6f}\n")
    
    print(f"Created dataset with {num_train} training and {num_val} validation images")
    print(f"Class names: {', '.join(class_names)}")
    print(f"Dataset directory: {base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a synthetic dataset for object detection')
    parser.add_argument('--output_dir', type=str, default='./example_dataset', 
                      help='Output directory for the dataset')
    parser.add_argument('--num_train', type=int, default=100, 
                      help='Number of training images')
    parser.add_argument('--num_val', type=int, default=20, 
                      help='Number of validation images')
    parser.add_argument('--img_size', type=int, default=64, 
                      help='Image size (square)')
    parser.add_argument('--num_classes', type=int, default=3, 
                      help='Number of object classes')
    
    args = parser.parse_args()
    
    create_dataset(
        args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        img_size=args.img_size,
        num_classes=args.num_classes
    ) 