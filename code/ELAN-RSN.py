from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, \
    GlobalMaxPool2D, Reshape,Add,Multiply,Activation,Concatenate,Conv1D
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import math

def channel_black(inputs, ratio=4):
    channel = inputs.shape[-1]
    # 设置共享全连接层
    share_dense1 = Dense(channel // ratio,activation='relu')
    share_dense2 = Dense(channel)
    #输入全局最大池化以及全局平均池化
    x_maxp = GlobalMaxPool2D()(inputs)
    x_avep = GlobalAveragePooling2D()(inputs)
    #
    x_maxp = Reshape(target_shape=(1, 1, channel))(x_maxp)
    x_avep = Reshape(target_shape=(1, 1, channel))(x_avep)
    #输入全连接
    x_maxp=share_dense1(x_maxp)
    x_maxp=share_dense2(x_maxp)
    x_avep=share_dense1(x_avep)
    x_avep=share_dense2(x_avep)

    x=Add()([x_maxp,x_avep])
    x=Activation('sigmoid')(x)
    return Multiply()([inputs,x])

def channel_max(x):
    x=K.max(x,axis=-1,keepdims=True)
    return x
def channel_ave(x):
    x=K.mean(x,axis=-1,keepdims=True)
    return x
def spatial_block(inputs):
    kernel_size = 7
    # 每一个特征点的通道上取最大值和平均值
    x_max=Lambda(channel_max)(inputs)
    x_ave=Lambda(channel_ave)(inputs)
    x=Concatenate()([x_max,x_ave])
    #调整通道数为1
    x = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(x)



    # x=Conv2D(1,[1,1])(x)
    x=Activation('sigmoid')(x)
    return Multiply()([inputs, x])



def DRSN_block(inputs, out_channels, downsample_strides=1):

    in_channels = inputs.shape[-1]

    residual = tf.keras.layers.BatchNormalization()(inputs)
    residual = tf.keras.layers.Activation('relu')(residual)
    residual = tf.keras.layers.Conv2D(out_channels, 3, padding='same')(residual)

    residual_abs = tf.abs(residual)  # 求绝对值
    abs_mean = tf.keras.layers.GlobalMaxPool2D()(residual_abs)

#ECANet
    kernel_size = int(abs((math.log(in_channels, 2) + 1) / 2))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    # c,1
    scales = Reshape([-1, 1])(abs_mean)
    scales = Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False)(scales)
    scales = Activation('sigmoid')(scales)
    # 1,1,c
    scales = Reshape([1, 1, -1])(scales)

    thres = tf.keras.layers.multiply([abs_mean, scales])  # 获得阈值

    sub = tf.keras.layers.subtract([residual_abs, thres])
    zeros = tf.keras.layers.subtract([sub, sub])
    n_sub = tf.keras.layers.maximum([sub, zeros])
    residual = tf.keras.layers.multiply([tf.sign(residual), n_sub])

    # out_channels = residual.shape[-1]
    # if in_channels == out_channels:
    #     identity = tf.keras.layers.Conv2D(out_channels, 1, strides=(downsample_strides, downsample_strides),
    #                                       padding='same')(inputs)

    residual = tf.keras.layers.add([residual, inputs])

    return residual




