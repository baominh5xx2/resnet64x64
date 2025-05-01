# ResNet Object Detection for Small Images (64x64)

This project implements an object detection model based on ResNet architecture, optimized for small 64x64 images. The model is designed to work with YOLO-formatted datasets.

## Features

- Custom ResNet-based detection model for small images
- Support for YOLO dataset format
- Complete training, evaluation and inference pipeline
- Visualization tools for detection results

## Requirements

- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd resnet_small_images

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib
```

## Dataset Structure

The dataset should follow the YOLO format:

```
dataset/
  ├── train/
  │   ├── images/
  │   │   ├── image1.jpg
  │   │   ├── image2.jpg
  │   │   └── ...
  │   ├── labels/
  │   │   ├── image1.txt
  │   │   ├── image2.txt
  │   │   └── ...
  │   └── classes.txt
  └── val/
      ├── images/
      │   ├── image1.jpg
      │   ├── image2.jpg
      │   └── ...
      ├── labels/
      │   ├── image1.txt
      │   ├── image2.txt
      │   └── ...
      └── classes.txt
```

Each label file should follow the YOLO format:
- One object per line
- Each line: `<class_id> <x_center> <y_center> <width> <height>`
- All values are normalized to [0, 1]

## Training

To train the model:

```bash
python train.py --data_path /path/to/dataset \
                --output_dir ./output \
                --img_size 64 \
                --grid_size 8 \
                --batch_size 16 \
                --epochs 50 \
                --evaluate
```

### Training Arguments

- `--data_path`: Path to the dataset directory (required)
- `--output_dir`: Directory to save models and results (default: ./output)
- `--img_size`: Image size (square) (default: 64)
- `--grid_size`: Detection grid size (default: 8)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of epochs to train (default: 50)
- `--workers`: Number of worker processes for data loading (default: 4)
- `--evaluate`: Evaluate model after training (optional flag)

## Inference

For inference on a single image or directory of images:

```bash
# Single image inference
python inference.py --model_path ./output/model_best.h5 \
                   --image_path /path/to/image.jpg \
                   --class_file /path/to/classes.txt \
                   --output_dir ./inference_results

# Batch inference on directory
python inference.py --model_path ./output/model_best.h5 \
                   --image_dir /path/to/images/ \
                   --class_file /path/to/classes.txt \
                   --output_dir ./inference_results
```

### Inference Arguments

- `--model_path`: Path to the trained model .h5 file (required)
- `--image_path`: Path to a single image for inference
- `--image_dir`: Path to directory of images for batch inference
- `--class_file`: Path to the class names file (optional)
- `--output_dir`: Directory to save inference results (default: ./inference_results)
- `--img_size`: Image size for the model (default: 64)
- `--grid_size`: Detection grid size (default: 8)
- `--confidence`: Confidence threshold for detections (default: 0.3)

## Model Architecture

The model is based on ResNet architecture with these key components:

1. Input layer for 64x64 RGB images
2. Efficient initial layer with depthwise separable convolution
3. Series of ResNet blocks with increasing filters (64 → 128 → 256 → 512)
4. Detection head that outputs predictions in a grid format (8x8 by default)
5. Each grid cell predicts:
   - Bounding box coordinates (x, y, width, height)
   - Objectness score
   - Class probabilities

## Customization

- Adjust `grid_size` for different detection granularity
- Modify the model architecture in `detection_model.py`
- Customize data augmentation in `dataset.py`
- Adjust loss function weights in `build_detection_model()` function

## License

[MIT License](LICENSE) 