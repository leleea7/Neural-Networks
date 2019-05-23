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

    '''if 'landmarks' in tasks:
        output_landmarks = tf.keras.layers.Dense(10)(layer)
    else:
        output_landmarks = None
    if 'gender' in tasks:
        output_gender = tf.keras.layers.Dense(2)(layer)
    else:
        output_gender = None
    if 'smile' in tasks:
        output_smile = tf.keras.layers.Dense(2)(layer)
    else:
        output_smile = None
    if 'glasses' in tasks:
        output_glasses = tf.keras.layers.Dense(2)(layer)
    else:
        output_glasses = None
    if 'head_pose' in tasks:
        output_head_pose = tf.keras.layers.Dense(5)(layer)
    else:
        output_head_pose = None'''

    for task in tasks:
        if task == 'Landmarks':
            output[task] = tf.keras.layers.Dense(10)(layer)
        else:
            output[task] = tf.keras.layers.Dense(2)(layer)

    return output


def Xception(images, num_classes_gender, num_classes_smile, num_classes_glasses, num_classes_head_pose):

    regularizer = tf.keras.regularizers.l2()

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

    '''with tf.name_scope('output_landmarks'):
        layer_landmarks = tf.keras.layers.Conv2D(num_classes_landmarks, (3, 3), padding='same')(layer)
        layer_landmarks = tf.keras.layers.GlobalAveragePooling2D()(layer_landmarks)
        output_landmarks = tf.keras.layers.Activation('softmax')(layer_landmarks)'''

    with tf.name_scope('output_gender'):
        layer_gender = tf.keras.layers.Conv2D(num_classes_gender, (3, 3), padding='same')(layer)
        layer_gender = tf.keras.layers.GlobalAveragePooling2D()(layer_gender)
        output_gender = tf.keras.layers.Activation('softmax')(layer_gender)

    with tf.name_scope('output_smile'):
        layer_smile = tf.keras.layers.Conv2D(num_classes_smile, (3, 3), padding='same')(layer)
        layer_smile = tf.keras.layers.GlobalAveragePooling2D()(layer_smile)
        output_smile = tf.keras.layers.Activation('softmax')(layer_smile)

    with tf.name_scope('output_glasses'):
        layer_glasses = tf.keras.layers.Conv2D(num_classes_glasses, (3, 3), padding='same')(layer)
        layer_glasses = tf.keras.layers.GlobalAveragePooling2D()(layer_glasses)
        output_glasses = tf.keras.layers.Activation('softmax')(layer_glasses)

    with tf.name_scope('output_head_pose'):
        layer_head_pose = tf.keras.layers.Conv2D(num_classes_head_pose, (3, 3), padding='same')(layer)
        layer_head_pose = tf.keras.layers.GlobalAveragePooling2D()(layer_head_pose)
        output_head_pose = tf.keras.layers.Activation('softmax')(layer_head_pose)

        '''with tf.name_scope('output'):
        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(100)(layer)
        layer = tf.keras.layers.Activation('relu')(layer)'''

    return output_gender, output_smile, output_glasses, output_head_pose
    #return layer

