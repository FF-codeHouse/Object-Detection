import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import Conv2D,Layer

class SpectralNorm(tf.keras.constraints.Constraint):
    def __init__(self, n_iter=5):  # 超参数
        self.n_iter = n_iter

    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))
        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)
            u = tf.matmul(w, v)
            u /= tf.norm(u)
        spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a=True)
        return input_weights / spec_norm

class Self_Attention(Layer):
    def __init__(self):
        super(Self_Attention, self).__init__()

    def build(self, input_shape):
        n ,h ,w ,c = input_shape
        self.n_feats = h * w
        self.conv_theta = Conv2D( c, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Theta')
        self.conv_phi = Conv2D( c, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Phi')
        self.conv_g = Conv2D( c, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_g')
        self.conv_attn_g = Conv2D( c, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_AttnG')
        self.sigma = self.add_weight(shape=[1], initializer='zeros', trainable=True, name='sigma')
    def call(self, x):
        n, h, w, c = x.shape
        theta = self.conv_theta(x)
        theta = tf.reshape(theta, (-1, self.n_feats, theta.shape[-1]))
        phi = self.conv_phi(x)
        phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding='VALID')
        phi = tf.reshape(phi, (-1, self.n_feats//4, phi.shape[-1]))
        g = self.conv_g(x)
        g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding='VALID')
        g = tf.reshape(g, (-1, self.n_feats//4, g.shape[-1]))
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        print(attn)
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, (-1, h, w, attn_g.shape[-1]))
        attn_g = self.conv_attn_g(attn_g)
        output = x + self.sigma * attn_g
        return output

