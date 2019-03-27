from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary as bv
from HierarchicalTransformer import HierarchicalTransformer
from Decoder import Decoder
import matplotlib.pyplot as plt
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

if __name__ == "__main__":

    # Text parameters #
    document_max_sents = 30 #30
    summary_max_sents = 4 #4
    document_max_words_per_sent = 25 #25
    summary_max_words_per_sent = 15 #15
    topk = 3
    max_vocabulary = 200000 #200000 # El mejor 12811646411 es con 200k, ahora estoy probando a doblar el vocabulario a ver como afecta
    test_path = "./cnndailymail-unanonymized/test.csv"
    model_path = "./cnndailymail-unanonymized/best-models/transformer_12822646433256_PESENT_NOPEWORDS_200kvocab-00051-0.39637-0.94112.hdf5"
    vocabulary_path = "./cnndailymail-unanonymized/vocab.vb"
    output_file = "hts_pruebas.out"

    # Hierarchical Transformer Parameters #
    dropout_word_input = 0.
    dropout_sent_input = 0.
    dropout_word_output = 0.
    dropout_sent_output = 0.
    # El mejor modelo (simple_best) es con pe_words y pe_sentences True
    pe_words = False #True
    pe_sentences = True #True
    label_smoothing = 0.1
    embedding_dims = 128 #128
    n_word_encoders = 2 #1
    n_sentence_encoders = 2 #1
    attention_word_dims = 64 #64
    attention_sentence_dims = 64 #64
    n_w_heads = 3 #1
    n_s_heads = 3 #1
    dim_h = 256 #256
    output_word_encoder_dims = [embedding_dims for i in range(n_word_encoders)]
    output_sentence_encoder_dims = [embedding_dims for i in range(n_sentence_encoders)]
    word_attention_dims = [attention_word_dims for i in range(n_word_encoders)]
    sentence_attention_dims = [attention_sentence_dims for i in range(n_sentence_encoders)]
    n_word_heads = [n_w_heads for i in range(n_word_encoders)]
    n_sentence_heads = [n_s_heads for i in range(n_sentence_encoders)]

    vocabulary = bv.load_vocab(vocabulary_path, max_vocabulary)
    sp = StringProcessing(vocabulary)

    x_ts_articles, x_ts_summaries = sp.load_csv_samples(test_path)

    ht = HierarchicalTransformer(max_vocabulary = max_vocabulary + 2,
                                 document_max_sents = document_max_sents,
                                 summary_max_sents = summary_max_sents,
                                 document_max_words_per_sent = document_max_words_per_sent,
                                 summary_max_words_per_sent = summary_max_words_per_sent,
                                 embedding_dims = embedding_dims,
                                 output_word_encoder_dims = output_word_encoder_dims,
                                 output_sentence_encoder_dims = output_sentence_encoder_dims,
                                 word_attention_dims = word_attention_dims,
                                 sentence_attention_dims = sentence_attention_dims,
                                 n_word_heads = n_word_heads,
                                 n_sentence_heads = n_sentence_heads,
                                 dropout_word_input = dropout_word_input,
                                 dropout_sent_input = dropout_sent_input,
                                 dropout_word_output = dropout_word_output,
                                 dropout_sent_output = dropout_sent_output,
                                 pe_sentences = pe_sentences,
                                 pe_words = pe_words, dim_h = dim_h)

    ht.build()
    print(ht.model.summary())
    ht.compile(ht.model)
    ht.load(ht.model, model_path)

    decoder = Decoder(ht.attn_model, document_max_sents,
                      document_max_words_per_sent, sp, embedding_dims)

    # Continuar con Decoder._decode_sample
    decoder._decode_samples(x_ts_articles, x_ts_summaries,
                            topk, output_file,
                            average_encoders = False,
                            selected_encoder = -1,
                            visualization = True,
                            rows_visualization = 1,
                            cols_visualization = 3)
