import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def non_max_suppression(boxes, scores, threshold=0.5, max_boxes=10):
    """Apply Non-Maximum Suppression to remove overlapping boxes"""
    selected_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=max_boxes,
        iou_threshold=threshold
    )
    
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    
    return selected_boxes.numpy(), selected_scores.numpy(), selected_indices.numpy()

def process_predictions(predictions, grid_size=8, confidence_threshold=0.5, nms_threshold=0.5, num_classes=1):
    """Process model predictions to get bounding boxes in image coordinates"""
    batch_size = predictions.shape[0]
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for b in range(batch_size):
        boxes = []
        scores = []
        class_ids = []
        
        for row in range(grid_size):
            for col in range(grid_size):
                # Get prediction for this grid cell
                cell_pred = predictions[b, row, col]
                
                # Extract bounding box coordinates
                x_cell, y_cell, w_cell, h_cell = cell_pred[0:4]
                objectness = cell_pred[4]
                
                # Filter by objectness confidence
                if objectness < confidence_threshold:
                    continue
                
                # Get class predictions
                class_probs = cell_pred[5:5+num_classes]
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]
                
                # Combined score (objectness * class probability)
                score = objectness * class_score
                
                # Filter by final confidence
                if score < confidence_threshold:
                    continue
                
                # Convert cell coordinates to image coordinates (0-1 range)
                x_center = (col + x_cell) / grid_size
                y_center = (row + y_cell) / grid_size
                w = w_cell / grid_size
                h = h_cell / grid_size
                
                # Convert to [x_min, y_min, x_max, y_max] format
                x_min = max(0, x_center - w/2)
                y_min = max(0, y_center - h/2)
                x_max = min(1, x_center + w/2)
                y_max = min(1, y_center + h/2)
                
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(score)
                class_ids.append(class_id)
        
        # Apply non-max suppression if we have any boxes
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            class_ids = np.array(class_ids, dtype=np.int32)
            
            # Apply NMS
            nms_boxes, nms_scores, nms_indices = non_max_suppression(
                boxes, scores, nms_threshold
            )
            
            nms_classes = class_ids[nms_indices]
            
            all_boxes.append(nms_boxes)
            all_scores.append(nms_scores)
            all_classes.append(nms_classes)
        else:
            all_boxes.append(np.array([]))
            all_scores.append(np.array([]))
            all_classes.append(np.array([]))
    
    return all_boxes, all_scores, all_classes

def draw_boxes(image, boxes, scores, class_ids, class_names=None, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on the image"""
    # Convert normalized coordinates to pixel coordinates
    h, w, _ = image.shape
    
    # Make a copy to avoid modifying the original
    img_copy = image.copy()
    
    for i, box in enumerate(boxes):
        # Convert normalized coordinates to pixel coordinates
        x_min, y_min, x_max, y_max = box
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)
        
        # Draw the box
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Draw label
        score = scores[i]
        class_id = class_ids[i]
        label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw text background
        cv2.rectangle(
            img_copy, 
            (x_min, y_min - text_height - 5), 
            (x_min + text_width, y_min), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            img_copy, 
            label, 
            (x_min, y_min - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1, 
            cv2.LINE_AA
        )
    
    return img_copy

def plot_detection_results(images, all_boxes, all_scores, all_classes, class_names=None, figsize=(12, 12)):
    """Plot detection results for multiple images using matplotlib"""
    num_images = len(images)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_images):
        if i < len(images):
            img = images[i]
            boxes = all_boxes[i]
            scores = all_scores[i]
            classes = all_classes[i]
            
            # Display image
            axes[i].imshow(img)
            
            # Draw bounding boxes
            for j, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                
                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (x_min, y_min), 
                    width, 
                    height, 
                    linewidth=2, 
                    edgecolor='r', 
                    facecolor='none'
                )
                
                # Add the patch to the Axes
                axes[i].add_patch(rect)
                
                # Add label
                score = scores[j]
                class_id = classes[j]
                label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
                
                axes[i].text(
                    x_min, 
                    y_min, 
                    label,
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5)
                )
            
            # Hide axis
            axes[i].axis('off')
        else:
            # Hide unused subplots
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig 