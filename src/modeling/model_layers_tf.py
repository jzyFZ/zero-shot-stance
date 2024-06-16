import tensorflow as tf
import math


class ScaledDotProductAttentionTF(tf.keras.layers.Layer):
    def __init__(self, input_dim, use_cuda=False):
        super(ScaledDotProductAttentionTF, self).__init__()
        self.input_dim = input_dim
        self.scale = math.sqrt(2 * self.input_dim)

    def call(self, inputs, query):
        # inputs = (B, L, 2 * H), query = (B, 2 * H), last_hidden=(B, 2 * H)
        sim = tf.einsum('blh,bh->bl', inputs, query) / self.scale  # (B, L)
        att_weights = tf.nn.softmax(sim, axis=1)  # (B, L)
        context_vec = tf.einsum('blh,bl->bh', inputs, att_weights)  # (B, 2 * H)
        return context_vec
