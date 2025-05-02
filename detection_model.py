import tensorflow as tf
from tensorflow.keras import layers, models
from resnet import ResNetBlock, ResNetBuilder
# Import ANCHORS và NUM_ANCHORS từ utils
from utils import ANCHORS, NUM_ANCHORS

class ResNetYOLODetection:
    # Thêm num_anchors và anchors vào init
    def __init__(self, input_shape=(64, 64, 3), grid_size=8, num_classes=1, num_anchors=NUM_ANCHORS, anchors=ANCHORS, activation='swish', dropout_rate=0.25):
        self.input_shape = input_shape
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors # Lưu số lượng anchors
        self.anchors = tf.constant(anchors, dtype=tf.float32) # Lưu anchors dưới dạng tensor
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Output dimensions per anchor: 5 + num_classes (xywh, obj, class_probs)
        self.output_dims_per_anchor = 5 + self.num_classes
        # Total output dimensions per grid cell
        self.output_dims = self.num_anchors * self.output_dims_per_anchor

        # Calculate downsample factor needed
        self.downsample_factor = input_shape[0] // grid_size
        if input_shape[0] % grid_size != 0:
            print(f"Warning: input_shape[0] ({input_shape[0]}) not divisible by grid_size ({grid_size}). Feature map size might not match.")

    def build(self):
        inputs = layers.Input(shape=self.input_shape)

        # Backbone (ResNet)
        # Initial Conv
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=self.activation)(inputs)
        x = layers.BatchNormalization()(x)

        # ResNet Blocks
        downsamples_needed = int(tf.math.log(float(self.downsample_factor)) / tf.math.log(2.0))
        num_filters = 64
        for i in range(downsamples_needed):
            x = ResNetBlock(num_filters, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)
            num_filters *= 2
        # Add a few more blocks without downsampling
        for _ in range(2): # Add more blocks if needed for deeper features
             x = ResNetBlock(num_filters, downsample=False, activation=self.activation, dropout_rate=self.dropout_rate)(x)

        # --- Feature map size check and adjustment ---
        current_grid_size = self.input_shape[0] // (2**downsamples_needed)
        if current_grid_size > self.grid_size:
            pool_factor = current_grid_size // self.grid_size
            print(f"Downsampling further with AveragePooling2D by factor {pool_factor}")
            x = layers.AveragePooling2D(pool_size=pool_factor, strides=pool_factor, padding='same')(x)
        elif current_grid_size < self.grid_size:
            upsample_factor = self.grid_size // current_grid_size
            print(f"Upsampling features with UpSampling2D by factor {upsample_factor}")
            x = layers.UpSampling2D(size=upsample_factor, interpolation='bilinear')(x)
        # ---------------------------------------------

        # Detection head
        x = layers.Conv2D(512, kernel_size=3, padding='same', activation=self.activation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x) # Increased dropout before final layer

        # Lớp Conv cuối cùng với số kênh mới
        raw_detection_output = layers.Conv2D(
            self.output_dims, # Số kênh mới = num_anchors * (5 + num_classes)
            kernel_size=1,
            activation=None, # Linear activation
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.005), # L2 regularization
            name='raw_detection_output'
        )(x)

        print(f"Raw output shape: {raw_detection_output.shape}") # Sẽ là (None, S, S, num_anchors * (5+C))

        # Reshape output để tách chiều anchor
        # Shape: (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
        reshaped_output = layers.Reshape(
            (self.grid_size, self.grid_size, self.num_anchors, self.output_dims_per_anchor),
            name='reshaped_output'
        )(raw_detection_output)

        print(f"Reshaped output shape: {reshaped_output.shape}")

        # Apply activations to different parts of the prediction
        # Box coordinates (x, y) -> sigmoid [0, 1] relative to cell
        pred_xy = layers.Activation('sigmoid', name='pred_xy')(reshaped_output[..., 0:2])
        # Box dimensions (w, h) -> linear (sẽ được xử lý bằng exp trong loss/postprocess)
        # No activation here, raw output represents log(w/anchor_w), log(h/anchor_h)
        pred_wh = layers.Lambda(lambda t: t, name='pred_wh')(reshaped_output[..., 2:4])
        # Objectness score -> sigmoid [0, 1] probability
        pred_obj = layers.Activation('sigmoid', name='pred_obj')(reshaped_output[..., 4:5])
        # Class probabilities -> sigmoid (cho phép multi-label hoặc đơn giản hơn)
        pred_class = layers.Activation('sigmoid', name='pred_class')(reshaped_output[..., 5:])

        # Ghép nối lại output đã được activate
        detection_output = layers.Concatenate(axis=-1, name='detection_output')([
            pred_xy, pred_wh, pred_obj, pred_class
        ])

        print(f"Final activated output shape: {detection_output.shape}") # Sẽ là (None, S, S, num_anchors, 5+C)

        model = models.Model(inputs, detection_output)
        return model

# Hàm loss mới
def build_detection_model(input_shape=(64, 64, 3), grid_size=8, num_classes=1, num_anchors=NUM_ANCHORS, anchors=ANCHORS, learning_rate=0.001):
    """Convenience function to build and compile the detection model with anchors"""
    model_builder = ResNetYOLODetection(
        input_shape=input_shape,
        grid_size=grid_size,
        num_classes=num_classes,
        num_anchors=num_anchors,
        anchors=anchors
    )

    model = model_builder.build()

    # Lấy anchors và grid_size để dùng trong hàm loss
    _anchors = tf.constant(anchors, dtype=tf.float32)
    _grid_size = grid_size
    _num_classes = num_classes
    _ignore_thresh = 0.5 # Ngưỡng IoU để bỏ qua anchor khi tính no_obj_loss

    # Custom loss function for object detection with anchors
    @tf.function # Thêm tf.function để tối ưu hóa
    def detection_loss_with_anchors(y_true, y_pred):
        # y_true, y_pred shape: [batch, grid_size, grid_size, num_anchors, 5+num_classes]

        # Mask để xác định anchor nào chịu trách nhiệm cho object (objectness = 1 trong y_true)
        obj_mask = y_true[..., 4:5] # Shape: [batch, S, S, num_anchors, 1]
        # Mask cho background (objectness = 0)
        no_obj_mask = 1.0 - obj_mask

        # --- Coordinate Loss (chỉ tính cho anchor chịu trách nhiệm) ---
        true_xy = y_true[..., 0:2]
        pred_xy = y_pred[..., 0:2]
        # Sử dụng Binary Crossentropy cho xy loss (ổn định hơn MSE trong khoảng 0-1)
        xy_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(true_xy, pred_xy) * obj_mask)

        true_wh_rel = y_true[..., 2:4] # w, h đã được mã hóa log trong dataset
        pred_wh_rel = y_pred[..., 2:4]
        # Sử dụng MSE loss cho wh loss (vì giá trị là log, không bị giới hạn)
        # Scale loss để cân bằng với xy_loss (có thể điều chỉnh)
        wh_loss = tf.reduce_sum(tf.square(true_wh_rel - pred_wh_rel) * obj_mask) * 0.5


        # --- Objectness Loss ---
        pred_obj = y_pred[..., 4:5]

        # Loss cho anchor chịu trách nhiệm (target là 1)
        obj_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(obj_mask, pred_obj) * obj_mask)

        # Loss cho background anchors (target là 0)
        # Cần tính IoU giữa predicted box và true box để xác định anchor nào cần ignore
        # (Phần này phức tạp, tạm thời dùng no_obj_mask đơn giản để đảm bảo chạy được)
        # TODO: Implement ignore threshold based on IoU
        no_obj_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(obj_mask, pred_obj) * no_obj_mask)


        # --- Class Loss (chỉ tính cho anchor chịu trách nhiệm) ---
        true_class = y_true[..., 5:]
        pred_class = y_pred[..., 5:]
        # Dùng binary_crossentropy vì đã dùng sigmoid activation cho class
        class_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(true_class, pred_class) * obj_mask)

        # --- Tổng hợp Loss ---
        # Chuẩn hóa loss bằng số lượng object hoặc batch size
        num_objects = tf.maximum(tf.reduce_sum(obj_mask), 1.0) # Số lượng anchor chịu trách nhiệm
        batch_size_f = tf.cast(tf.shape(y_true)[0], tf.float32)

        # Chuẩn hóa bằng batch size để ổn định hơn khi số object thay đổi nhiều
        xy_loss /= batch_size_f
        wh_loss /= batch_size_f
        obj_loss /= batch_size_f
        no_obj_loss /= batch_size_f
        class_loss /= batch_size_f

        # Trọng số loss (có thể cần tinh chỉnh)
        lambda_coord = 5.0
        lambda_noobj = 0.5
        lambda_obj = 1.0
        lambda_class = 1.0

        total_loss = (lambda_coord * (xy_loss + wh_loss) +
                      lambda_obj * obj_loss +
                      lambda_noobj * no_obj_loss +
                      lambda_class * class_loss)

        # Thêm regularization loss của model
        total_loss += tf.reduce_sum(model.losses)

        return total_loss

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=detection_loss_with_anchors
    )

    return model