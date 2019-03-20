from keras import backend as K
from keras.layers import Layer
from numpy import sqrt

class SelfAttention(Layer):

    def __init__(self, d, **kwargs):
        self.d = d
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wq = self.add_weight(shape=(input_shape[-1], self.d),
                                  name="Wq", initializer="glorot_uniform",
                                  trainable=True)

        self.wk = self.add_weight(shape=(input_shape[-1], self.d),
                                  name="Wk", initializer="glorot_uniform",
                                  trainable=True)

        self.wv = self.add_weight(shape=(input_shape[-1], self.d),
                                  name="Wv", initializer="glorot_uniform",
                                  trainable=True)

        super(SelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):
        q = K.dot(x, self.wq)
        k = K.dot(x, self.wk)
        v = K.dot(x, self.wv)
        attn = K.softmax(K.batch_dot(q, K.permute_dimensions(k, (0, 2, 1))) / sqrt(self.d))
        return [K.batch_dot(attn, v), attn]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1], self.d), (input_shape[1], input_shape[1])]