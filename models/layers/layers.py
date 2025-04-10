import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Conv2D, BatchNormalization, Add, Normalization


class MatrixMulti(tf.keras.layers.Layer):
    def __call__(self, x1, x2):
        return tf.linalg.matmul(x1, x2, transpose_b=True)

class ElementMulti(tf.keras.layers.Layer):
    def __call__(self, x1, x2):
        return tf.math.multiply(x1, x2)

class PixelShuffle(tf.keras.layers.Layer):
    def __call__(self, x, scale):
        return tf.nn.depth_to_space(x, scale)

class Rescale_(tf.keras.layers.Layer):
    def __call__(self, x, a=0, b=1):
        cast_input = tf.cast(x, dtype=tf.float32)  # e.g. MNIST is integer
        input_min = tf.reduce_min(cast_input, axis=[1, 2])  # result B x C
        input_max = tf.reduce_max(cast_input, axis=[1, 2])
        ex_min = tf.expand_dims(input_min, axis=1)  # put back inner dimensions
        ex_max = tf.expand_dims(input_max, axis=1)
        ex_min = tf.expand_dims(ex_min, axis=1)  # one at a time - better way?
        ex_max = tf.expand_dims(ex_max, axis=1)  # Now Bx1x1xC
        input_range = tf.subtract(ex_max, ex_min)
        floored = tf.subtract(cast_input, ex_min)  # broadcast
        scale_input = tf.divide(floored, input_range)
        return scale_input


# Self Calibrated Convolution with Pixel Attention
class SCPA(tf.keras.layers.Layer):
    def __call__(self, x, filters=16):
        x1 = Conv2D(filters, 1, strides=s, padding='same')(x)  # 1x1 conv
        x1 = Conv2D(filters, 3, strides=s, padding='same')(x1)  # 3x3 conv

        x2 = Conv2D(filters, 1, strides=s, padding='same')(x)  # 1x1 conv
        x3 = Conv2D(filters, 3, strides=s, padding='same')(x2)  # 3x3 conv
        x2 = Conv2D(filters, 1, strides=s, padding='same')(x2)  # 1x1 conv

        x2 = MatrixMulti()(x2, x3)  # Matrix multiplication
        x2 = Conv2D(filters, 3, strides=s, padding='same')(x2)  # 3x3 conv

        x1 = ElementMulti()(x1, x2)  # Element-wise multiplication
        x1 = Conv2D(filters, 1, strides=s, padding='same')(x1)  # 1x1 conv
        x = Add()([x, x1])  # Element-wise addition (skip all)
        return x

# Gated Dconv Feed-Forward Network
class GDFN(tf.keras.layers.Layer):
    def __call__(self, x, filters=16):
        x1 = Normalization()(x)

        x2 = Conv2D(filters, 1, strides=s, padding='same')(x1)  # 1x1 conv
        x2 = Conv2D(filters, 3, strides=s, padding='same')(x2)  # 3x3 conv

        x1 = Conv2D(filters, 1, strides=s, padding='same')(x1)  # 1x1 conv
        x1 = Conv2D(filters, 3, strides=s, padding='same')(x1)  # 3x3 conv
        x1 = Activation('gelu')(x1)  # GeLU activation

        x1 = ElementMulti()(x1, x2)  # Element-wise multiplication
        x1 = Conv2D(filters, 1, strides=s, padding='same')(x1)  # 1x1 conv

        x = Add()([x, x1])  # Element-wise addition (skip all)
        return x

# Enhanced Residual Block
class ERB(tf.keras.layers.Layer):
    def __call__(x, filters=16, s=1):
        x2 = Conv2D(filters, 1, strides=s, padding='same')(x)  # 1x1 conv
        x3 = Conv2D(filters, 3, strides=s, padding='same')(x2)  # 3x3 conv
        x3 = Add()([x2, x3])  # Element-wise addition (skip 3x3)
        x3 = Conv2D(filters, 1, strides=s, padding='same')(x3)  # 1x1 conv
        x = Add()([x, x3])  # Element-wise addition (skip all)
        return x

# High-Frequuency Attention Block
class HFAB(tf.keras.layers.Layer):
    def __call__(x, filters=16):  # 20
        x1 = BatchNormalization()(x)  # Batch normalization
        x1 = Conv2D(filters, 3, strides=s, padding='same')(x1)  # 3x3 conv
        x1 = Activation('relu')(x1)  # ReLU activation
        x1 = ERB(x1, filters)  # ERB block
        x1 = Activation('relu')(x1)  # ReLU activation
        x1 = BatchNormalization()(x1)  # Batch normalization
        x1 = Conv2D(filters, 3, strides=s, padding='same')(x1)  # 3x3 conv
        x1 = BatchNormalization()(x1)  # Batch normalization
        x1 = Activation('sigmoid')(x1)  # Sigmoid activation
        # x  = MatrixMulti()(x, x1)                             # Matrix multiplication NOT USED
        x = ElementMulti()(x, x1)  # Matrix multiplication
        return x