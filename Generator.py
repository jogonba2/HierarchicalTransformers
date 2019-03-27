from random import randint
from StringProcessing import StringProcessing
from Utils import Utils
import numpy as np


class Generator:

    def __init__(self, articles, summaries,
                 document_max_sents,
                 summary_max_sents,
                 document_max_words_per_sent,
                 summary_max_words_per_sent,
                 vocabulary, pos_pairs, neg_pairs,
                 embedding_dims, label_smoothing = 0.1,
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
        self.d_model = embedding_dims
        self.word_delimiter = word_delimiter
        self.sentence_delimiter = sentence_delimiter
        self.label_smoothing = label_smoothing
        self.sp = StringProcessing(self.vocabulary)

    def generator(self):
        matrix_encodings_words_article = Utils.precompute_word_pos_encodings(self.document_max_sents, self.document_max_words_per_sent, self.d_model)
        matrix_encodings_words_summary = Utils.precompute_word_pos_encodings(self.summary_max_sents, self.summary_max_words_per_sent, self.d_model)
        matrix_encodings_sents_article = Utils.precompute_sent_pos_encodings(self.document_max_sents, self.d_model)
        matrix_encodings_sents_summary = Utils.precompute_sent_pos_encodings(self.summary_max_sents, self.d_model)

        while True:
            batch_x1, batch_x2, batch_y = [], [], []
            batch_word_x1_mask, batch_sent_x1_mask = [], []
            batch_word_x2_mask, batch_sent_x2_mask = [], []
            batch_word_x1_pe, batch_sent_x1_pe = [], []
            batch_word_x2_pe, batch_sent_x2_pe = [], []

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

                word_mask_article = (doc_repr>0).astype("int")
                word_mask_summary = (summ_repr>0).astype("int")
                sent_mask_article = (word_mask_article.sum(axis=1)>0).astype("int")
                sent_mask_summary = (word_mask_summary.sum(axis=1)>0).astype("int")
                pos_encodings_word_article = Utils.build_pe_word_encodings(matrix_encodings_words_article, word_mask_article)
                pos_encodings_word_summary = Utils.build_pe_word_encodings(matrix_encodings_words_summary, word_mask_summary)
                pos_encodings_sent_article = Utils.build_pe_sent_encodings(matrix_encodings_sents_article, sent_mask_article)
                pos_encodings_sent_summary = Utils.build_pe_sent_encodings(matrix_encodings_sents_summary, sent_mask_summary)
                batch_x1.append(doc_repr)
                batch_x2.append(summ_repr)
                batch_word_x1_mask.append(word_mask_article)
                batch_sent_x1_mask.append(sent_mask_article)
                batch_word_x2_mask.append(word_mask_summary)
                batch_sent_x2_mask.append(sent_mask_summary)
                batch_word_x1_pe.append(pos_encodings_word_article)
                batch_sent_x1_pe.append(pos_encodings_sent_article)
                batch_word_x2_pe.append(pos_encodings_word_summary)
                batch_sent_x2_pe.append(pos_encodings_sent_summary)
                batch_y.append([self.label_smoothing, 1. - self.label_smoothing])

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


                word_mask_article = (doc_repr>0).astype("int")
                word_mask_summary = (summ_repr>0).astype("int")
                sent_mask_article = (word_mask_article.sum(axis=1)>0).astype("int")
                sent_mask_summary = (word_mask_summary.sum(axis=1)>0).astype("int")
                pos_encodings_word_article = Utils.build_pe_word_encodings(matrix_encodings_words_article, word_mask_article)
                pos_encodings_word_summary = Utils.build_pe_word_encodings(matrix_encodings_words_summary, word_mask_summary)
                pos_encodings_sent_article = Utils.build_pe_sent_encodings(matrix_encodings_sents_article, sent_mask_article)
                pos_encodings_sent_summary = Utils.build_pe_sent_encodings(matrix_encodings_sents_summary, sent_mask_summary)

                batch_x1.append(doc_repr)
                batch_x2.append(summ_repr)
                batch_word_x1_mask.append(word_mask_article)
                batch_sent_x1_mask.append(sent_mask_article)
                batch_word_x2_mask.append(word_mask_summary)
                batch_sent_x2_mask.append(sent_mask_summary)
                batch_word_x1_pe.append(pos_encodings_word_article)
                batch_sent_x1_pe.append(pos_encodings_sent_article)
                batch_word_x2_pe.append(pos_encodings_word_summary)
                batch_sent_x2_pe.append(pos_encodings_sent_summary)
                batch_y.append([1. - self.label_smoothing, self.label_smoothing])

            batch_x1 = np.array(batch_x1)
            batch_x2 = np.array(batch_x2)
            batch_word_x1_mask = np.array(batch_word_x1_mask)
            batch_sent_x1_mask = np.array(batch_sent_x1_mask)
            batch_word_x2_mask = np.array(batch_word_x2_mask)
            batch_sent_x2_mask = np.array(batch_sent_x2_mask)
            batch_word_x1_pe = np.array(batch_word_x1_pe)
            batch_sent_x1_pe = np.array(batch_sent_x1_pe)
            batch_word_x2_pe = np.array(batch_word_x2_pe)
            batch_sent_x2_pe = np.array(batch_sent_x2_pe)
            batch_y = np.array(batch_y)

            yield ([batch_x1, batch_word_x1_mask, batch_sent_x1_mask, batch_word_x1_pe, batch_sent_x1_pe,
                    batch_x2, batch_word_x2_mask, batch_sent_x2_mask, batch_word_x2_pe, batch_sent_x2_pe], batch_y)
