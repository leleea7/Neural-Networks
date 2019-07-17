import tensorflow as tf

def TCDCNv2(images, tasks):
    output = {}
    # shape(images) = (60, 60, 1)

    layer = tf.keras.layers.Conv2D(20, (5, 5), padding='same')(images)
    # shape(layer) = (56, 56, 20)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=1)(layer)
    # shape(layer) = (28, 28, 20)

    layer = tf.keras.layers.Conv2D(48, (5, 5), padding='same')(layer)
    # shape(layer) = (24, 24, 48)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=1)(layer)
    # shape(layer) = (12, 12, 48)

    layer = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(layer)
    # shape(layer) = (10, 10, 64)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=1)(layer)
    # shape(layer) = (5, 5, 64)

    layer = tf.keras.layers.Conv2D(80, (3, 3), padding='same')(layer)
    # shape(layer) = (3, 3, 80)
    layer = tf.keras.layers.Activation('relu')(layer)

    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(256)(layer)

    for task in tasks:
        if task == 'Landmarks':
            output[task] = tf.keras.layers.Dense(10)(layer)
        else:
            output[task] = tf.keras.layers.Dense(2)(layer)

    return output


def TCDCN(images, tasks):
    output = {}
    regularization = tf.keras.regularizers.l2(0.01)
    layer = tf.keras.layers.Conv2D(16, (5, 5), padding='same', kernel_regularizer=regularization)(images)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=2)(layer)

    layer = tf.keras.layers.Conv2D(48, (3, 3), padding='same', kernel_regularizer=regularization)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=2)(layer)

    layer = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularization)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=2)(layer)

    layer = tf.keras.layers.Conv2D(64, (2, 2), padding='same', kernel_regularizer=regularization)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Activation('relu')(layer)

    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(100)(layer)
    layer = tf.keras.layers.Activation('relu')(layer)

    for task in tasks:
        if task == 'Landmarks':
            output[task] = tf.keras.layers.Dense(10)(layer)
        else:
            output[task] = tf.keras.layers.Dense(2)(layer)

    return output