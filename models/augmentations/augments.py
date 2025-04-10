### TODO: FIX THIS
from tensorflow.keras.layers import Resizing, Rescaling, RandomFlip, RandomRotation, RandomCrop

resize_and_rescale = tf.keras.Sequential([
        RandomCrop(input_shape[0] // 1.3, input_shape[1] // 1.3, 3),
        Resizing(input_shape[0], input_shape[1]),
        Rescaling(1. / 255)], )

data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2), ])
