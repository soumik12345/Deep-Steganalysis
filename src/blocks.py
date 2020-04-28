import tensorflow as tf


def T1(input_tensor, n_filters, kernel_size=(3, 3)):
    y = tf.keras.layers.Conv2D(
        n_filters, kernel_size, padding='same'
    )(input_tensor)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    return y


def T2(input_tensor, n_filters, kernel_size=(3, 3)):
    y = T1(input_tensor, n_filters)
    y = tf.keras.layers.Conv2D(
        n_filters, kernel_size, padding='same'
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Add()([y, input_tensor])
    return y


def T3(input_tensor, n_filters, kernel_size=(3, 3)):
    y = T1(input_tensor, n_filters)
    y = tf.keras.layers.Conv2D(
        n_filters, kernel_size, padding='same'
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.AveragePooling2D(
        pool_size=(3, 3), strides=(2, 2)
    )(y)
    y_branch = tf.keras.layers.Conv2D(
        n_filters, kernel_size, strides=(2, 2)
    )(input_tensor)
    y_branch = tf.keras.layers.BatchNormalization()(y_branch)
    y = tf.keras.layers.Add()([y, y_branch])
    return y


def T4(input_tensor, n_filters, kernel_size=(3, 3)):
    y = T1(input_tensor, n_filters)
    y = tf.keras.layers.Conv2D(
        n_filters, kernel_size, padding='same'
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    return y
