from keras.layers import Layer, Dense, BatchNormalization, Add
from keras import backend as K
from MultiHeadAttention import MultiHeadAttention
from LayerNormalization import LayerNormalization

class SentenceEncoderBlock(Layer):

    def __init__(self, output_dim, attention_dim, n_heads, dropout = 0.3, **kwargs):
        self.output_dim = output_dim # Es la dimensión de salida del encoder después de las fc
        self.n_heads = n_heads
        self.attention_dim = attention_dim # Es la dimensión para dq/dk/dv de multihead attention
        self.activation = "relu"
        self.dropout = dropout
        super(SentenceEncoderBlock, self).__init__(**kwargs)

    def build(self, input_shape):

        # "Two linear transformations with a ReLU activation in between" #
        self.dense_1 = Dense(self.output_dim, activation=self.activation)
        self.dense_1.build(input_shape)
        self._trainable_weights += self.dense_1.trainable_weights

        self.dense_2 = Dense(self.output_dim)
        self.dense_2.build(input_shape)
        self._trainable_weights += self.dense_2.trainable_weights

        # MultiHeadAttention #
        self.multihead_attention = MultiHeadAttention(self.attention_dim, self.n_heads)
        self.multihead_attention.build(input_shape)
        self._trainable_weights += self.multihead_attention.trainable_weights

        # LayerNorm #
        self.layer_normalization = LayerNormalization()
        self.layer_normalization.build(input_shape)
        self._trainable_weights += self.layer_normalization.trainable_weights

        super(SentenceEncoderBlock, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):

        z, all_attns = self.multihead_attention(x)
        #z = K.dropout(z, self.dropout)
        xz = self.layer_normalization(x + z)
        h_xz = self.dense_1(xz)
        h_xz = self.dense_2(h_xz)
        #h_xz = K.dropout(h_xz, self.dropout)
        h_xz = self.layer_normalization(h_xz + xz)
        return [h_xz, all_attns]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1], self.output_dim),
                (input_shape[0], self.n_heads, input_shape[1], input_shape[1])]
