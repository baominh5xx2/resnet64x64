import tensorflow as tf
from tensorflow.keras import layers, models
from resnet import ResNetBlock, ResNetBuilder

class ResNetYOLODetection:
    def __init__(self, input_shape=(64, 64, 3), grid_size=8, num_classes=1, activation='swish', dropout_rate=0.25):
        self.input_shape = input_shape
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Total output dimensions per grid cell: 
        # 4 for bounding box (x, y, w, h)
        # 1 for objectness score
        # num_classes for class probabilities
        self.output_dims = 5 + num_classes
        
        # Calculate needed downsampling to reach required grid size
        self.downsample_factor = self.input_shape[0] // self.grid_size
        print(f"Input shape: {self.input_shape}, Grid size: {self.grid_size}, Downsample factor: {self.downsample_factor}")
    
    def build(self):
        # Use the ResNetBuilder for the backbone with modified output
        resnet_backbone = ResNetBuilder(
            input_shape=self.input_shape,
            num_classes=self.num_classes,  # This will be ignored as we'll modify the output
            activation=self.activation, 
            dropout_rate=self.dropout_rate
        )
        
        # Start with the backbone but don't use its output layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Efficient initial layer for small images
        x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(inputs)
        x = layers.Conv2D(32, kernel_size=1, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        
        # Print current size
        print(f"After initial layer: {x.shape}")
        
        # Tính toán số lần cần downsample để đạt được kích thước grid
        # Ví dụ: từ 64x64 -> 8x8 cần 3 lần downsample (stride=2)
        # Với 320x320 -> 8x8 cần 4-5 lần
        
        # Residual blocks với downsampling được điều chỉnh theo kích thước đầu vào
        # Đổi block đầu tiên từ downsample=False thành downsample=True để tạo projection kênh từ 32->64
        x = ResNetBlock(64, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        print(f"After ResNet block 1: {x.shape}")
        
        # Block thứ hai giữ nguyên kích thước
        x = ResNetBlock(64, downsample=False, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        print(f"After ResNet block 2: {x.shape}")
        
        # Bắt đầu áp dụng downsample để giảm kích thước
        # Tính toán số lần downsample còn lại để đạt được grid_size
        current_size = self.input_shape[0]  # Ban đầu là 64 hoặc 320
        downsamples_needed = 0
        
        while current_size > self.grid_size:
            current_size = current_size // 2
            downsamples_needed += 1
        
        # Đã dùng 1 lần downsample ở block đầu
        downsamples_needed = max(0, downsamples_needed - 1)
        print(f"Downsamples needed: {downsamples_needed}")
        
        # Áp dụng các lần downsample còn lại
        filters = 128
        for i in range(downsamples_needed):
            x = ResNetBlock(filters, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)
            print(f"After ResNet block {i+3}: {x.shape}")
            filters = min(512, filters * 2)  # Tăng số filter, tối đa là 512
        
        # Kiểm tra kích thước đầu ra
        current_shape = x.shape[1]  # Lấy kích thước height
        if current_shape != self.grid_size:
            print(f"WARNING: Current output shape {current_shape} doesn't match desired grid size {self.grid_size}")
            # Áp dụng thêm upsampling hoặc pooling nếu cần
            if current_shape < self.grid_size:
                # Upsampling nếu output quá nhỏ
                scale_factor = self.grid_size // current_shape
                x = layers.UpSampling2D(size=(scale_factor, scale_factor))(x)
                print(f"Applied upsampling: {x.shape}")
            elif current_shape > self.grid_size:
                # Downsampling nếu output quá lớn
                pool_size = current_shape // self.grid_size
                x = layers.AveragePooling2D(pool_size=(pool_size, pool_size))(x)
                print(f"Applied pooling: {x.shape}")
        
        # Detection head - predict bounding boxes, objectness and class probabilities
        x = layers.Conv2D(512, kernel_size=3, padding='same', activation=self.activation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)  # Thêm dropout
        
        raw_detection_output = layers.Conv2D(
            self.output_dims,
            kernel_size=1,
            activation=None,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),  # Thêm L2 regularization
            name='raw_detection_output'
        )(x)

        print(f"Raw output shape: {raw_detection_output.shape}")

        # Apply activations to different parts of the output using Lambda layers if needed
        # Box coordinates (x, y) -> sigmoid [0, 1] relative to cell
        pred_xy = layers.Activation('sigmoid', name='pred_xy')(raw_detection_output[..., 0:2])
        # Box dimensions (w, h) -> linear (or exp for positivity, but linear is simpler with MSE)
        pred_wh = layers.Lambda(lambda x: x, name='pred_wh')(raw_detection_output[..., 2:4])
        # Objectness score -> sigmoid [0, 1] probability
        pred_obj = layers.Activation('sigmoid', name='pred_obj')(raw_detection_output[..., 4:5])
        # Class probabilities -> softmax across classes
        pred_class = layers.Activation('softmax', name='pred_class')(raw_detection_output[..., 5:])

        # IMPORTANT: Use Concatenate layer from Keras instead of tf.concat
        detection_output = layers.Concatenate(axis=-1, name='detection_output')([
            pred_xy, pred_wh, pred_obj, pred_class
        ])

        print(f"Final activated output shape: {detection_output.shape}")

        # Create model
        model = models.Model(inputs, detection_output)
        return model

def build_detection_model(input_shape=(64, 64, 3), grid_size=8, num_classes=1):
    """Convenience function to build and compile the detection model"""
    model_builder = ResNetYOLODetection(
        input_shape=input_shape,
        grid_size=grid_size,
        num_classes=num_classes
    )
    
    model = model_builder.build()
    
    # Custom loss function for object detection
    def detection_loss(y_true, y_pred):
        # Debugging info - chỉ in khi chạy lần đầu
        if not hasattr(detection_loss, 'shape_printed'):
            print(f"y_true shape: {y_true.shape}")
            print(f"y_pred shape: {y_pred.shape}")
            detection_loss.shape_printed = True
        
        # Extract components from prediction tensors
        # y_true and y_pred shapes: [batch, grid_size, grid_size, 5+num_classes]
        
        # Box coordinates and dimensions
        pred_xy = y_pred[..., 0:2]
        pred_wh = y_pred[..., 2:4]
        
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]
        
        # Objectness scores (already sigmoid activated)
        pred_obj = y_pred[..., 4:5]
        true_obj = y_true[..., 4:5]
        
        # Class probabilities (already softmax activated)
        pred_class = y_pred[..., 5:]
        true_class = y_true[..., 5:]
        
        # Calculate coordinate loss (using mean squared error)
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * true_obj) / tf.maximum(tf.reduce_sum(true_obj), 1.0)
        wh_loss = tf.reduce_sum(tf.square(tf.sqrt(true_wh + 1e-6) - tf.sqrt(tf.abs(pred_wh) + 1e-6)) * true_obj) / tf.maximum(tf.reduce_sum(true_obj), 1.0)
        
        # Calculate objectness loss (binary cross-entropy)
        obj_loss_raw = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        
        # Expand dims để phù hợp với kích thước của true_obj
        obj_loss_raw = tf.expand_dims(obj_loss_raw, axis=-1)  # Shape: [batch, grid, grid, 1]
        
        # Giờ phép nhân sẽ hoạt động đúng
        obj_loss = tf.reduce_sum(obj_loss_raw * true_obj) / tf.maximum(tf.reduce_sum(true_obj), 1.0)
        
        # Penalize background predictions more
        no_obj_mask = 1.0 - true_obj  # Shape: [batch, grid, grid, 1]
        
        # Tương tự cho no_obj_loss_raw
        no_obj_loss_raw_expanded = tf.expand_dims(tf.keras.losses.binary_crossentropy(true_obj, pred_obj), axis=-1)
        no_obj_loss_raw = no_obj_loss_raw_expanded * no_obj_mask
        no_obj_loss = tf.reduce_sum(no_obj_loss_raw) / tf.maximum(tf.reduce_sum(no_obj_mask), 1.0)
        
        # Sửa lỗi class loss - đảm bảo kích thước tương thích
        # Tính categorical_crossentropy giữa true_class và pred_class
        class_loss_per_cell = tf.keras.losses.categorical_crossentropy(true_class, pred_class)  # Shape: [batch, grid, grid]
        
        # Chuyển class_loss_per_cell thành [batch, grid, grid, 1] để nhân với true_obj
        class_loss_per_cell = tf.expand_dims(class_loss_per_cell, axis=-1)  # Shape: [batch, grid, grid, 1]
        
        # Nhân trực tiếp với true_obj (không cần squeeze)
        class_loss = tf.reduce_sum(class_loss_per_cell * true_obj) / tf.maximum(tf.reduce_sum(true_obj), 1.0)
        
        # Total loss with weighting factors
        lambda_coord = 5.0
        lambda_noobj = 0.5
        lambda_obj = 1.0
        lambda_class = 1.0
        
        total_loss = (lambda_coord * (xy_loss + wh_loss) +
                      lambda_obj * obj_loss +
                      lambda_noobj * no_obj_loss +
                      lambda_class * class_loss)
        
        return total_loss
    
    # Compile model with custom loss and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=detection_loss
    )
    
    return model