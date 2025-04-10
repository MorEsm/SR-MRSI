import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Activation, Dense, Concatenate, Conv2D

from layers.layers import *

def model_name(input_shape, filters=32, residuals=5, s=1, scale=4):
    input = Input(input_shape, )

    #input = resize_and_rescale(input)
    #input = data_augmentation(input)

    x = Conv2D(filters, 1, strides=s, padding='same')(input)  # 1x1 conv
    x = Conv2D(filters, 3, strides=s, padding='same')(x)  # 3x3 conv

    x1 = x
    for i in range(n_residuals):
        x1 = SCPA(x1)  # scpa block
        x1 = HFAB(x1)  # hfab block

    x1 = Add()([x, x1])

    x1 = Conv2D(filters, 1, strides=s, padding='same')(x1)  # 1x1 conv
    x1 = Conv2D(filters, 3, strides=s, padding='same')(x1)  # 3x3 conv

    x = Concatenate()([x, x1])  # Concatenate

    x1 = GDFN(x, filters=filters * 2)  # gdfn block
    x1 = Conv2D(filters, 3, strides=s, padding='same')(x1)  # 3x3 conv

    x = Concatenate()([x, x1])

    x1 = Conv2D(filters, 1, strides=s, padding='same')(x1)  # 1x1 conv
    x1 = Conv2D(filters, 3, strides=s, padding='same')(x1)  # 3x3 conv

    x = Concatenate()([x, x1])  # Concatenate

    # pixel shuffle
    if scale != 1:
        x = PixelShuffle()(x, scale=scale)  # Pixel shuffle layer
    x = Conv2D(input_shape[-1], 1, strides=s, padding='same')(x)  # 1x1 conv

    x = Activation('sigmoid')(x)  # To normalize
    # x = Rescale_()(x)

    output = x

    model = Model(input, output)
    return model