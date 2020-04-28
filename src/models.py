from .blocks import *
import tensorflow as tf


def SRNet(input_shape=(512, 512, 3)):
    input_tensor = tf.keras.Input(shape=input_shape)
    y = T1(input_tensor, 64)
    y = T1(y, 16)
    for i in range(5):
        y = T2(y, 16)
    for n_filters in [16, 64, 128, 256]:
        y = T3(y, n_filters)
    y = T4(y, 512)
    y = tf.keras.layers.Dense(128)(y)
    output_tensor = tf.keras.layers.Dense(128, activation='softmax')(y)
    return tf.keras.Model(input_tensor, output_tensor)
