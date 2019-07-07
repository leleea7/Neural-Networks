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
    layer = tf.keras.layers.Conv2D(16, (5, 5), padding='same')(images)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=2)(layer)

    layer = tf.keras.layers.Conv2D(48, (3, 3), padding='same')(layer)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=2)(layer)

    layer = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(layer)
    layer = tf.keras.layers.Activation('relu')(layer)
    layer = tf.keras.layers.MaxPool2D((2, 2), strides=2)(layer)

    layer = tf.keras.layers.Conv2D(64, (2, 2), padding='same')(layer)
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


def Xception(images, tasks):

    regularizer = tf.keras.regularizers.l2()
    output = {}

    with tf.name_scope('base'):
        layer = tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularizer, use_bias=False)(
            images)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularizer, use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)

    with tf.name_scope('module1'):
        residual = tf.keras.layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('module2'):
        residual = tf.keras.layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('module3'):
        residual = tf.keras.layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('module4'):
        residual = tf.keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('output'):

        for task in tasks:
            if task == 'Landmarks':
                output[task] = tf.keras.layers.Conv2D(10, (3, 3), padding='same')(layer)
            else:
                output[task] = tf.keras.layers.Conv2D(2, (3, 3), padding='same')(layer)
            output[task] = tf.keras.layers.GlobalAveragePooling2D()(output[task])

    return output
