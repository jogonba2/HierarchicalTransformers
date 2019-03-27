from keras.layers import (Input, Embedding,  TimeDistributed, GlobalMaxPooling1D)
from keras.models import Model
from StringProcessing import StringProcessing
from keras import backend as K
from keras.layers import Layer
import numpy as np

class MyMasking(Layer):

    def __init__(self, **kwargs):
        super(MyMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyMasking, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):
        mask = K.expand_dims(mask, 2)
        mask = K.repeat_elements(mask, 100, -1)
        return x * mask

    def compute_output_shape(self, input_shape):
        return input_shape

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

def build():
    input_article = Input(shape=(6, 5))
    input_summary = Input(shape=(6, 5))

    # Máscaras a nivel de palabras #
    word_mask_article = Input(shape=(6, 5))
    word_mask_summary = Input(shape=(6, 5))

    # Máscaras a nivel de frases #
    sent_mask_article = Input(shape=(6,))
    sent_mask_summary = Input(shape=(6,))

    # Pos encodings #
    pos_encoding_word_article = Input(shape=(6, 5, 100))
    pos_encoding_word_summary = Input(shape=(6, 5, 100))

    # Pos encodings #
    pos_encoding_sent_article = Input(shape=(6, 100))
    pos_encoding_sent_summary = Input(shape=(6, 100))


    embedding = Embedding(9, 100)

    article_embedding = embedding(input_article)
    summary_embedding = embedding(input_summary)

    ep_article = TimeDistributed(MyMasking())(article_embedding, mask = word_mask_article)

    ep_article = TimeDistributed(PositionalEncoding())(ep_article,
                                                       mask = pos_encoding_word_article)

    #ep_article = TimeDistributed(GlobalMaxPooling1D())(ep_article)

    #ep_article = PositionalEncoding()(ep_article, mask = pos_encoding_sent_article)

    model = Model(inputs=[input_article, word_mask_article, sent_mask_article, pos_encoding_word_article, pos_encoding_sent_article,
                          input_summary, word_mask_summary, sent_mask_summary, pos_encoding_word_summary, pos_encoding_sent_summary],
                  outputs=[article_embedding, ep_article])

    return model


model = build()
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
#print(model.summary())

document_max_sents = 6
summary_max_sents = 6
document_max_words_per_sent = 5
summary_max_words_per_sent = 5
x = ["hola amigo mio . como estas . bien y tu . mal",
     "hola amigo mio . como estas . bien y tu"]

y = ["hola amigo mio . como estas . bien y tu",
     "hola amigo mio . como estas . bien y tu"]
vocab = {"hola" : 1, "amigo" : 2, "mio": 3, "como":4, "estas":5, "bien":6, "y":7, "tu":8}
sp = StringProcessing(vocab)

rx = sp.represent_documents(x, document_max_sents,
                           document_max_words_per_sent)

ry = sp.represent_documents(y, document_max_sents,
                           document_max_words_per_sent)

    def roll_zeropad(a, shift, axis=None):
        a = np.asanyarray(a)
        if shift == 0: return a
        if axis is None:
            n = a.size
            reshape = True
        else:
            n = a.shape[axis]
            reshape = False
        if np.abs(shift) > n:
            res = np.zeros_like(a)
        elif shift < 0:
            shift += n
            zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
            res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
        else:
            zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
            res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
        if reshape:
            return res.reshape(a.shape)
        else:
            return res


    def precompute_word_pos_encodings(max_sents, max_words_sents, d_model):
        r = np.zeros((max_sents, max_words_sents, d_model))
        for i in range(max_sents):
            # Positional encoding de cada frase por separado (orden a nivel de palabras dentro de frases)
            for pos in range(max_words_sents):
                for k in range(d_model):
                    if k % 2 == 0:
                        r[i][pos][k] = np.sin(pos / (10000 ** ((2 * k) / d_model)))
                    else:
                        r[i][pos][k] = np.cos(pos / (10000 ** ((2 * k) / d_model)))
        return r

    def precompute_sent_pos_encodings(max_sents, d_model):
        r = np.zeros((max_sents, d_model))
        for pos in range(max_sents):
            # Positional encoding sobre todas las frases (orden a nivel de frase)
            for k in range(d_model):
                if k % 2 == 0:
                    r[pos][k] = np.sin(pos / (10000 ** ((2 * k) / d_model)))
                else:
                    r[pos][k] = np.cos(pos / (10000 ** ((2 * k) / d_model)))
        return r


    def build_pe_word_encodings(pos_encodings, mask):
        rolls = (1. - mask).sum(axis=1).astype("int")
        r = np.copy(pos_encodings)
        for i in range(len(pos_encodings)):
            r[i] = roll_zeropad(r[i], rolls[i], axis=0)
        return r

    def build_pe_sent_encodings(pos_encodings, mask):
        rolls = (1. - mask).sum(axis=0).astype("int")
        r = np.copy(pos_encodings)
        r = roll_zeropad(r, rolls, axis=0)
        return r

document_max_sents = 6
summary_max_sents = 6
document_max_words_per_sent = 5
summary_max_words_per_sent = 5

matrix_encodings_words_article = precompute_word_pos_encodings(document_max_sents, document_max_words_per_sent, 100)
matrix_encodings_words_summary = precompute_word_pos_encodings(summary_max_sents, summary_max_words_per_sent, 100)
matrix_encodings_sents_article = precompute_sent_pos_encodings(document_max_sents, 100)
matrix_encodings_sents_summary = precompute_sent_pos_encodings(summary_max_sents, 100)

word_masks_article = np.array([(rx[i]>0).astype("int") for i in range(len(rx))])
word_masks_summary = np.array([(ry[i]>0).astype("int") for i in range(len(ry))])
sent_masks_article = np.array([(word_masks_article[i].sum(axis=1)>0).astype("int") for i in range(len(word_masks_article))])
sent_masks_summary = np.array([(word_masks_summary[i].sum(axis=1)>0).astype("int") for i in range(len(word_masks_summary))])

pos_encodings_words_article = np.array([build_pe_word_encodings(matrix_encodings_words_article, v) for v in word_masks_article])
pos_encodings_words_summary = np.array([build_pe_word_encodings(matrix_encodings_words_summary, v) for v in word_masks_summary])
pos_encodings_sents_article = np.array([build_pe_sent_encodings(matrix_encodings_sents_article, v) for v in sent_masks_article])
pos_encodings_sents_summary = np.array([build_pe_sent_encodings(matrix_encodings_sents_summary, v) for v in sent_masks_summary])

p = model.predict([rx, word_masks_article, sent_masks_article, pos_encodings_words_article, pos_encodings_sents_article,
                   ry, word_masks_summary, sent_masks_summary, pos_encodings_words_summary, pos_encodings_sents_summary])

# OUTPUT\BATCH #
embs = p[0][0]
padd = p[1][0]

print("EMBEDDINGS\n")
print(embs)
print("\n\n\nPADDING MASK\n")
print(padd)
