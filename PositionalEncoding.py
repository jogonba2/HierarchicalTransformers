from keras import backend as K
from keras.layers import Layer


class PositionalEncoding(Layer):

    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    # mask son los positinal encodings #
    def call(self, x, mask=None):
        return x + mask

    def compute_output_shape(self, input_shape):
        return input_shape
