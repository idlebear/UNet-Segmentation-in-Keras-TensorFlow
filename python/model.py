
# Original model definition taken from
# https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow with additional
# modifications for brevity and reuse

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate
from tensorflow.keras.layers import UpSampling2D, Input
from tensorflow.keras import Model

# ## Model Definition
#
# ### Convolutional Blocks

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, activation="relu")(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, activation="relu")(concat)
    c = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, activation="relu")(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, activation="relu")(c)
    return c


# ### Model Details
def UNet(image_size, filters=[16, 32, 64, 128, 256]):

    inputs = Input(image_size+(3,))

    connects = []
    p = inputs

    # down one side of the U
    for f in filters[:-1]:
        c, p = down_block(p, f)
        connects.append(c)

    # last filter is for the bottleneck
    p = bottleneck(p, filters[-1])

    # and up the other
    filters = filters[::-1][1:]
    connects = connects[::-1]
    for f, c in zip(filters, connects):
        p = up_block(p, c, f)

    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(p)
    model = Model(inputs, outputs)

    return model
