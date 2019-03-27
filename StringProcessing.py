from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


class StringProcessing:

    def __init__(self, vocabulary):
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1
        self.vocabulary = vocabulary

    @staticmethod
    def load_csv_samples(csv_path):
        aux_samples = []
        for chunk in pd.read_csv(csv_path, sep='\s*\t\s*',
                                 lineterminator="\n", chunksize=20000,
                                 engine="python", encoding="utf8"):
            aux_samples.append(chunk)

        csv_samples = pd.concat(aux_samples, axis=0)
        del aux_samples

        return csv_samples["TEXT"].tolist(), csv_samples["SUMMARY"].tolist()

    def represent_sentence(self, sentence, word_delimiter = " "):
        if not sentence: return None
        sent = sentence.split(word_delimiter)
        r = []
        for w in sent:
            if w in self.vocabulary:
                r.append(self.vocabulary[w])
            else:
                r.append(self.UNK_TOKEN)
        return r

    def represent_documents(self, documents, max_sentences,
                            max_words_per_sentence,
                            word_delimiter = " ",
                            sentence_delimiter = " . "):

        return [self.represent_document(document, max_sentences,
                                        max_words_per_sentence,
                                        word_delimiter,
                                        sentence_delimiter)
                for document in documents]

    # Padding PRE, Truncating POST #
    def represent_document(self, document, max_sentences,
                           max_words_per_sentence,
                           word_delimiter = " ",
                           sentence_delimiter = " . "):

        if not document:
            return np.zeros((max_sentences, max_words_per_sentence), dtype="int32")

        sentences = document.split(sentence_delimiter)

        repr_sentences = [self.represent_sentence(sentence, word_delimiter) if sentence \
                          is not None else [] for sentence in sentences]

        repr_sentences = pad_sequences(repr_sentences, maxlen = max_words_per_sentence,
                                       dtype="int32", padding="pre", truncating="post",
                                       value = 0)

        repr_document = pad_sequences([repr_sentences], maxlen = max_sentences,
                                       dtype="int32", padding="pre", truncating="post",
                                       value = 0)[0]

        return repr_document