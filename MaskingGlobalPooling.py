from keras.layers import Layer
from keras import backend as K
import tensorflow as tf

class MaskingGlobalPooling(Layer):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskingGlobalPooling, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0, 2, 1])
            x = x * mask
        return K.sum(x, axis=1) / K.sum(mask, axis=1)

    def get_output_shape_for(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[2]