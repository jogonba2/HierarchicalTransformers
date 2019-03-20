from random import randint
from StringProcessing import StringProcessing
import numpy as np


class Generator:

    def __init__(self, articles, summaries,
                 document_max_sents,
                 summary_max_sents,
                 document_max_words_per_sent,
                 summary_max_words_per_sent,
                 vocabulary, pos_pairs, neg_pairs,
                 word_delimiter = " ", sentence_delimiter = " . "):

        self.articles = articles
        self.summaries = summaries
        self.n_articles = len(self.articles)
        self.document_max_sents = document_max_sents
        self.summary_max_sents = summary_max_sents
        self.document_max_words_per_sent = document_max_words_per_sent
        self.summary_max_words_per_sent = summary_max_words_per_sent
        self.vocabulary = vocabulary
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs
        self.word_delimiter = word_delimiter
        self.sentence_delimiter = sentence_delimiter
        self.sp = StringProcessing(self.vocabulary)

    def generator(self):

        while True:
            batch_x1, batch_x2, batch_y = [], [], []
            for cpos in range(self.pos_pairs):
                index = randint(0, self.n_articles - 1)
                doc = self.articles[index]
                summ = self.summaries[index]
                if not doc or not sum:
                    cpos -= 1
                    continue

                doc_repr = self.sp.represent_document(doc, self.document_max_sents,
                                                      self.document_max_words_per_sent,
                                                      self.word_delimiter,
                                                      self.sentence_delimiter)

                summ_repr = self.sp.represent_document(summ, self.summary_max_sents,
                                                      self.summary_max_words_per_sent,
                                                      self.word_delimiter,
                                                      self.sentence_delimiter)

                batch_x1.append(doc_repr)
                batch_x2.append(summ_repr)
                batch_y.append([0, 1])


            for cneg in range(self.neg_pairs):
                doc_index = randint(0, self.n_articles - 1)
                summ_index = randint(0, self.n_articles - 1)
                while doc_index == summ_index: summ_index = randint(0, self.n_articles - 1)
                doc = self.articles[doc_index]
                summ = self.summaries[summ_index]
                if not doc or not summ:
                    cneg -= 1
                    continue

                doc_repr = self.sp.represent_document(doc, self.document_max_sents,
                                                      self.document_max_words_per_sent,
                                                      self.word_delimiter,
                                                      self.sentence_delimiter)

                summ_repr = self.sp.represent_document(summ, self.summary_max_sents,
                                                      self.summary_max_words_per_sent,
                                                      self.word_delimiter,
                                                      self.sentence_delimiter)

                batch_x1.append(doc_repr)
                batch_x2.append(summ_repr)
                batch_y.append([1, 0])

            batch_x1 = np.array(batch_x1)
            batch_x2 = np.array(batch_x2)
            batch_y = np.array(batch_y)

            yield ([batch_x1, batch_x2], batch_y)