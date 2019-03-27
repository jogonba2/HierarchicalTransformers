from keras import backend as K
from keras.layers import Layer
import numpy as np


class MyMasking(Layer):

    def __init__(self, **kwargs):
        super(MyMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d = input_shape[-1]
        super(MyMasking, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):
        mask = K.expand_dims(mask, 2)
        mask = K.repeat_elements(mask, self.d, -1)
        return x * mask

    def compute_output_shape(self, input_shape):
        return input_shape
