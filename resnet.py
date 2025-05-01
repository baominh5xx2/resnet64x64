import tensorflow as tf
from tensorflow.keras import layers, models

class ResNetBlock(tf.keras.Model):
    def __init__(self, filters, downsample=False, activation='swish', dropout_rate=0.0):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.downsample = downsample
        if downsample:
            self.downsample_conv = layers.Conv2D(filters, kernel_size=1, strides=2)
            self.downsample_bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.dropout:
            x = self.dropout(x, training=training)

        if self.downsample:
            residual = self.downsample_conv(inputs)
            residual = self.downsample_bn(residual, training=training)

        x += residual
        return self.activation(x)

class ResNetBuilder:
    def __init__(self, input_shape, num_classes, activation='swish', dropout_rate=0.25):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_rate = dropout_rate

    def build(self):
        inputs = layers.Input(shape=self.input_shape)

        # Efficient initial layer for small images
        x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(inputs)
        x = layers.Conv2D(32, kernel_size=1, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        # Residual blocks with increasing filters
        x = ResNetBlock(64, downsample=False, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        x = ResNetBlock(64, downsample=False, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        x = ResNetBlock(128, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        x = ResNetBlock(128, downsample=False, activation=self.activation, dropout_rate=self.dropout_rate)(x)
        x = ResNetBlock(256, downsample=True, activation=self.activation, dropout_rate=self.dropout_rate)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        return model

# Example usage
# resnet = ResNetBuilder(input_shape=(64, 64, 3), num_classes=200)
# model = resnet.build()
# model.summary()
