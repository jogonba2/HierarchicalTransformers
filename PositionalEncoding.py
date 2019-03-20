from keras.layers import Layer
import numpy as np

class PositionalEncoding(Layer):

    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.max_len = input_shape[1]
        self.d = input_shape[-1]
        self.pos_encodings = np.zeros((self.max_len, self.d))
        for i in range(self.max_len):
            for j in range(self.d):
                if j%2==0:
                    self.pos_encodings[i][j] = np.sin(i / (10000 ** ((2 * j) / self.d)))
                else:
                    self.pos_encodings[i][j] = np.cos(i / (10000 ** ((2 * j) / self.d)))

        super(PositionalEncoding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):
        return x + self.pos_encodings

    def compute_output_shape(self, input_shape):
        return input_shape