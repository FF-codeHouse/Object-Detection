import math
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import random_normal
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, DepthwiseConv2D, GlobalAveragePooling2D,
                          Lambda, Multiply, Reshape, Conv1D, multiply,Input)
# from MobileVIT import mobilevit_block
from Modul.Self_Attention import Self_Attention
from nets.backbone import DarknetConv2D_BN_Leaky
def eca_block(input_feature, b=1, gamma=2):
    channel = input_feature.shape[-1]
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    avg_pool = GlobalAveragePooling2D()(input_feature)

    x = Reshape((-1, 1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False, )(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1, 1, -1))(x)

    output = multiply([input_feature, x])
    return output

def slices(dw, n):
    return dw[:, :, :, :n]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _squeeze(inputs, hidden_channel, ratio, block_id, sub_block_id):
    # --------------------------------------------------------------------------#
    #   这是一个简单的注意力机制，和senet的结构类似
    # --------------------------------------------------------------------------#
    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, -1))(x)

    x = Conv2D(_make_divisible(hidden_channel / ratio, 4), (1, 1), strides=(1, 1), padding='same',
               kernel_initializer=random_normal(stddev=0.02),
               name="blocks." + str(block_id) + "." + str(sub_block_id) + ".se.conv_reduce")(x)
    x = Activation('relu')(x)

    x = Conv2D(hidden_channel, (1, 1), strides=(1, 1), padding='same', kernel_initializer=random_normal(stddev=0.02),
               name="blocks." + str(block_id) + "." + str(sub_block_id) + ".se.conv_expand")(x)
    x = Activation('hard_sigmoid')(x)

    x = Multiply()([inputs, x])  # inputs和x逐元素相乘
    return x


def _ghost_module(inputs, exp, ratio, kernel_size=1, dw_size=3, stride=1, relu=True):
    # --------------------------------------------------------------------------#
    #   ratio一般会指定成2
    #   这样才可以保证输出特征层的通道数，等于exp
    # --------------------------------------------------------------------------#
    output_channels = math.ceil(exp * 1.0 / ratio)

    # --------------------------------------------------------------------------#
    #   利用1x1卷积对我们输入进来的特征图进行一个通道的缩减，获得一个特征浓缩
    #   跨通道的特征提取
    # --------------------------------------------------------------------------#
    # x = Conv2D(output_channels, kernel_size, strides=stride, padding="same", use_bias=False,
    #            kernel_initializer=random_normal(stddev=0.02),
    #           )(
    #     inputs)
    # x = BatchNormalization(
    #     )(x)
    # if relu:
    #     x = Activation('relu')(x)
    x=DarknetConv2D_BN_Leaky(output_channels, kernel_size, weight_decay=5e-4)(inputs)
    print(x)
    # --------------------------------------------------------------------------#
    #   在获得特征浓缩之后，使用Transformer卷积，获得额外的特征图
    # --------------------------------------------------------------------------#
    dw = Self_Attention()(x)   #MobileViT block L=2
    dw = BatchNormalization(
        )(dw)
    if relu:
        dw = Activation('relu')(dw)
    print(dw)
    # --------------------------------------------------------------------------#
    #   将1x1卷积后的结果，和逐层卷积后的结果进行堆叠
    # --------------------------------------------------------------------------#
    x = Concatenate(axis=-1)([x, dw])
    x = Lambda(slices, arguments={'n': exp})(x)
    print(x)
    return x

def ghost_mobilevit(inputs, output_channel, hidden_channel, kernel, strides, ratio):
    input_shape = K.int_shape(inputs)

    # --------------------------------------------------------------------------#
    #   首先利用一个ghost模块进行特征提取
    #   此时指定的通道数会比较大，可以看作是逆残差结构
    # --------------------------------------------------------------------------#
    x = _ghost_module(inputs, hidden_channel, ratio)
    # 在两个_ghost_module之间添加一个ECA注意力机制
    x = eca_block(x)
    if strides > 1:
        # --------------------------------------------------------------------------#
        #   如果想要进行特征图的高宽压缩，则进行一个逐层卷积
        # --------------------------------------------------------------------------#
        x = DepthwiseConv2D(kernel, strides, padding='same', depth_multiplier=1, use_bias=False,
                            depthwise_initializer=random_normal(stddev=0.02))(x)
        x = BatchNormalization()(x)


    if strides == 1 and input_shape[-1] == output_channel:
        res = inputs
    else:
        res = DepthwiseConv2D(kernel, strides=strides, padding='same', depth_multiplier=1, use_bias=False,
                              depthwise_initializer=random_normal(stddev=0.02))(inputs)
        res = BatchNormalization()(res)
        res = Conv2D(output_channel, (1, 1), padding='same', strides=(1, 1), use_bias=False,
                     kernel_initializer=random_normal(stddev=0.02))(res)
        res = BatchNormalization()(res)
    x = Add()([res, x])

    return x
