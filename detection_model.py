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
        
        # Residual blocks with increasing filters (same as in ResNetBuilder)
        x = ResNetBlock(64, downsample=False, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        x = ResNetBlock(64, downsample=False, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        
        # After these blocks, feature map size is still the same as input (64x64)
        # We want to downsample to 8x8 grid for detection
        x = ResNetBlock(128, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)  # 32x32
        
        # Add more blocks with downsampling to reach 8x8 feature map
        x = ResNetBlock(256, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)  # 16x16
        
        # Last downsample to 8x8 (our grid size)
        x = ResNetBlock(512, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)  # 8x8
        
        # Detection head - predict bounding boxes, objectness and class probabilities
        # Output shape: [batch_size, grid_size, grid_size, output_dims]
        x = layers.Conv2D(512, kernel_size=3, padding='same', activation=self.activation)(x)
        detection_output = layers.Conv2D(
            self.output_dims, 
            kernel_size=1, 
            activation=None, 
            padding='same', 
            name='detection_output'
        )(x)
        
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
        # Extract components from prediction tensors
        # y_true and y_pred shapes: [batch, grid_size, grid_size, 5+num_classes]
        
        # Box coordinates and dimensions
        pred_xy = y_pred[..., 0:2]
        pred_wh = y_pred[..., 2:4]
        
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]
        
        # Objectness scores
        pred_obj = y_pred[..., 4:5]
        true_obj = y_true[..., 4:5]
        
        # Class probabilities
        pred_class = y_pred[..., 5:]
        true_class = y_true[..., 5:]
        
        # Calculate coordinate loss (using mean squared error)
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * true_obj) / tf.maximum(tf.reduce_sum(true_obj), 1)
        wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh) * true_obj) / tf.maximum(tf.reduce_sum(true_obj), 1)
        
        # Calculate objectness loss (binary cross-entropy)
        obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = tf.reduce_mean(obj_loss)
        
        # Calculate class loss (only where objects exist)
        class_loss = tf.keras.losses.categorical_crossentropy(true_class, pred_class) * true_obj[..., 0]
        class_loss = tf.reduce_sum(class_loss) / tf.maximum(tf.reduce_sum(true_obj), 1)
        
        # Total loss with weighting factors
        total_loss = 5 * xy_loss + 5 * wh_loss + obj_loss + class_loss
        
        return total_loss
    
    # Compile model with custom loss and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=detection_loss,
        metrics=['accuracy']  # Note: accuracy isn't very meaningful for detection tasks
    )
    
    return model 