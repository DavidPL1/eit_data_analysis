#!/usr/bin/env python3
import tensorflow as tf

def mlp1_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,16,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])

def mlp1_unreg_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,16,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])

def mlp2_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,16,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])

def mlp2_unreg_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,16,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])

def cnn2_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False, dilation_rate=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(12, activation='softmax')
    ], name="CNN2")


def cnn1_model(dropout=0.25):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False, dilation_rate=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False, dilation_rate=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False, dilation_rate=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(12, activation='softmax')
    ], name="CNN1")

def dual_branch_mlp2_model():
    InMeasure = tf.keras.Input(shape=(16, 16, 1))
    InRef = tf.keras.Input(shape=(16, 16, 1))

    x = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
    ])(InMeasure)

    y = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
    ])(InRef)

    x = tf.keras.backend.concatenate([x, y])

    x = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])(x)

    return tf.keras.Model(inputs=[InMeasure, InRef], outputs=x, name="dual-MLP2-dual-branch")


def single_branch_mlp2_model():
    InMeasure = tf.keras.Input(shape=(16, 16, 1))
    InRef = tf.keras.Input(shape=(16, 16, 1))

    x = tf.keras.backend.concatenate([InMeasure, InRef])
    x = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])(x)

    return tf.keras.Model(inputs=[InMeasure, InRef], outputs=x, name="dual-MLP2-single-branch")


def dual_branch_mlp1_model():
    InMeasure = tf.keras.Input(shape=(16, 16, 1))
    InRef = tf.keras.Input(shape=(16, 16, 1))

    x = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
    ])(InMeasure)

    y = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
    ])(InRef)

    x = tf.keras.backend.concatenate([x, y])

    x = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])(x)

    return tf.keras.Model(inputs=[InMeasure, InRef], outputs=x, name="dual-MLP1-dual-branch")

def single_branch_mlp1_model():
    InMeasure = tf.keras.Input(shape=(16, 16, 1))
    InRef = tf.keras.Input(shape=(16, 16, 1))

    x = tf.keras.backend.concatenate([InMeasure, InRef])
    x = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])(x)

    return tf.keras.Model(inputs=[InMeasure, InRef], outputs=x, name="dual-MLP1-single-branch")

def single_branch_input_cnn_template(
        activation_fn=tf.nn.relu,
        filters_1=64,
        filters_2=64,
        num_convs=2,
        dilation_rate=1,
        padding_1="same",
        padding_2="same",
        dropout_1=0.0,
        dropout_2=0.0,
        name="single_branch_in_template",
):
    InMeasure = tf.keras.Input(shape=(16, 16, 1))
    InRef = tf.keras.Input(shape=(16, 16, 1))

    x = tf.keras.backend.concatenate([InMeasure, InRef])

    x = tf.keras.layers.Conv2D(filters_1, kernel_size=3, strides=1, use_bias=False, padding=padding_1, dilation_rate=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_fn)(x)
    x = tf.keras.layers.Dropout(dropout_1)(x)

    for i in range(num_convs):
        x = tf.keras.layers.Conv2D(filters_2, kernel_size=3, strides=1, use_bias=False, padding=padding_2, dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation_fn)(x)
        x = tf.keras.layers.Dropout(dropout_2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(12, activation='softmax')(x)

    return tf.keras.Model(inputs=[InMeasure, InRef], outputs=x, name=name)


def dual_branch_input_cnn_template(
        dropout_1=0.0,
        dropout_2=0.0,
        activation_fn=tf.nn.relu,
        filters_1=64,
        filters_2=64,
        padding_1='same',
        padding_2='same',
        dilation_rate=3,
        num_convs=3,
        name="dual_branch_in_template"):
    InMeasure = tf.keras.Input(shape=(16, 16, 1))
    InRef = tf.keras.Input(shape=(16, 16, 1))

    x = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, 1)),
        tf.keras.layers.Conv2D(filters_1, kernel_size=3, strides=1, use_bias=False, padding=padding_1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation_fn),
        tf.keras.layers.Dropout(dropout_1),
    ])(InMeasure)

    y = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16, 16, 1)),
        tf.keras.layers.Conv2D(filters_1, kernel_size=3, strides=1, use_bias=False, padding=padding_1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation_fn),
        tf.keras.layers.Dropout(dropout_1),
    ])(InRef)

    x = tf.keras.backend.concatenate([x, y])

    for i in range(num_convs):
        x = tf.keras.layers.Conv2D(filters_2, kernel_size=3, strides=1, use_bias=False, padding=padding_2, dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation_fn)(x)
        x = tf.keras.layers.Dropout(dropout_2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(12, activation=tf.nn.softmax)(x)

    return tf.keras.Model(inputs=[InMeasure, InRef], outputs=x, name=name)


def dual_branch_cnn1_model():
    return dual_branch_input_cnn_template(num_convs=2, dilation_rate=2, name='dual-CNN1-dual-branch')


def single_branch_cnn1_model():
    return single_branch_input_cnn_template(num_convs=1, dilation_rate=2, name='dual-CNN1-single-branch')


def dual_branch_cnn3_model():
    return dual_branch_input_cnn_template(name='dual-CNN3-dual-branch')


def dual_branch_cnn3_dropout_model():
    return dual_branch_input_cnn_template(dropout_1=0.5, dropout_2=0.5, name="dual-CNN3-dual-branch-dropout")


def dual_branch_cnn3_dropout_2_model():
    return dual_branch_input_cnn_template(dropout_1=0.25, dropout_2=0.25, name="dual-CNN3-dual-branch-dropout-2")


def dual_branch_cnn4_model():
    return dual_branch_input_cnn_template(num_convs=1, dilation_rate=2, name="dual-CNN4-dual-branch")


def single_branch_cnn4_model():
    return single_branch_input_cnn_template(filters_1=128, num_convs=1, filters_2=128, dilation_rate=2, name='dual-CNN4-single-branch')


def dual_branch_cnn5_model():
    return dual_branch_input_cnn_template(filters_1=32, filters_2=32, dilation_rate=3, name='dual-CNN5-dual-branch')


def dual_branch_cnn6_model():
    return dual_branch_input_cnn_template(dilation_rate=1, num_convs=1, name='dual-CNN6-dual-branch')


def single_branch_cnn6_model():
    return single_branch_input_cnn_template(num_convs=1, name="dual-CNN6-single-branch")
