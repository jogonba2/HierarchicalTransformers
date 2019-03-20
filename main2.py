from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary as bv
from HierarchicalTransformer import HierarchicalTransformer
from keras.utils.np_utils import to_categorical
from Generator import Generator
import numpy as np

if __name__ == "__main__":

    # Text parameters #
    document_max_sents = 30
    summary_max_sents = 4
    document_max_words_per_sent = 25
    summary_max_words_per_sent = 15
    max_vocabulary = 10000
    train_path = "./mini_cnndm.csv"
    dev_path = "./mini_cnndm.csv"
    vocabulary_path = "./vocab.vb"

    # Training Parameters #
    pos_pairs = 32
    neg_pairs = 32
    epochs = 5
    steps_per_epoch = int(1000 / 64)

    # Hierarchical Transformer Parameters #
    embedding_dims = 32
    n_word_encoders = 3
    n_sentence_encoders = 3
    attention_word_dims = 32
    attention_sentence_dims = 32
    n_w_heads = 4
    n_s_heads = 4
    output_word_encoder_dims = [embedding_dims for i in range(n_word_encoders)]
    output_sentence_encoder_dims = [embedding_dims for i in range(n_sentence_encoders)]
    word_attention_dims = [attention_word_dims for i in range(n_word_encoders)]
    sentence_attention_dims = [attention_sentence_dims for i in range(n_sentence_encoders)]
    n_word_heads = [n_w_heads for i in range(n_word_encoders)]
    n_sentence_heads = [n_s_heads for i in range(n_sentence_encoders)]


    # Construir 1 sola vez el vocabulario #
    # bv.save_vocab(train_path, vocabulary_path)
    #############################################

    vocabulary = bv.load_vocab(vocabulary_path, max_vocabulary)
    sp = StringProcessing(vocabulary)

    x_tr_articles, x_tr_summaries = sp.load_csv_samples(train_path)

    tr_generator = Generator(x_tr_articles, x_tr_summaries,
                             document_max_sents, summary_max_sents,
                             document_max_words_per_sent, summary_max_words_per_sent,
                             vocabulary, pos_pairs, neg_pairs).generator()

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
                                 n_sentence_heads = n_sentence_heads
                                )

    ht.build()
    print(ht.model.summary())
    ht.compile(ht.model)



    ht.model.fit_generator(tr_generator, steps_per_epoch = steps_per_epoch,
                           epochs = epochs)

    ht.save(ht.model, "./second_version_weights.h5")

    ht.load(ht.model, "./second_version_weights.h5")

