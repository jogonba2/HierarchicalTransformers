import numpy as np

class Utils:

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def build_pe_word_encodings(pos_encodings, mask):
        rolls = (1. - mask).sum(axis=1).astype("int")
        r = np.copy(pos_encodings)
        for i in range(len(pos_encodings)):
            r[i] = Utils.roll_zeropad(r[i], rolls[i], axis=0)
        return r

    @staticmethod
    def build_pe_sent_encodings(pos_encodings, mask):
        rolls = (1. - mask).sum(axis=0).astype("int")
        r = np.copy(pos_encodings)
        r = Utils.roll_zeropad(r, rolls, axis=0)
        return r
