from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary as bv
from HierarchicalTransformer import HierarchicalTransformer
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from Generator import Generator
import numpy as np
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":

    # Text parameters #
    document_max_sents = 30 #30
    summary_max_sents = 4 #4
    document_max_words_per_sent = 25 #25
    summary_max_words_per_sent = 15 #15
    max_vocabulary = 200000 #200000 #100000
    train_path = "./cnndailymail-unanonymized/train.csv"
    dev_path = "./cnndailymail-unanonymized/dev.csv"
    vocabulary_path = "./cnndailymail-unanonymized/vocab.vb"

    # Training Parameters #
    pos_pairs = 32
    neg_pairs = 32
    epochs = 1000
    steps_per_epoch = 750
    validation_steps = 250
    path_models = "./cnndailymail-unanonymized/sht-models/"
    name_models = "transformer_12822646433256_PESENT_NOPEWORDS_200kvocab"
    # Hierarchical Transformer Parameters #
    # De momento, el que mejor he visto "decodificando" es (128, 2, 2, 64, 64, 8)
    # El modelo más simple de todos (128, 1, 1, 64, 64, 1, 1) es el que mejores resultados saca, y al decodificar también parece bueno.
    # Voy a entrenar dos modelos (128, 2, 2, 64, 64, 8) y (128, 1, 1, 64, 64, 1) 50 épocas cada uno, me los guardaré y probaré a decodificar y evaluar con ROUGE
    # Si los resultados no son catastróficos y el modelo más complejo todavía no ha convergido, lo entrenaré por más tiempo (el pequeño en unas 10 ya va por 96% accuracy...)
    # El mejor con diferencia es el más simple: 12811646411
    dropout_word_input = 0.
    dropout_sent_input = 0.
    dropout_word_output = 0.
    dropout_sent_output = 0.
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

    # Construir 1 sola vez el vocabulario #
    #bv.save_vocab(train_path, vocabulary_path)
    #############################################

    vocabulary = bv.load_vocab(vocabulary_path, max_vocabulary)
    sp = StringProcessing(vocabulary)

    x_tr_articles, x_tr_summaries = sp.load_csv_samples(train_path)
    x_dv_articles, x_dv_summaries = sp.load_csv_samples(dev_path)

    tr_generator = Generator(x_tr_articles, x_tr_summaries,
                             document_max_sents, summary_max_sents,
                             document_max_words_per_sent, summary_max_words_per_sent,
                             vocabulary, pos_pairs, neg_pairs, embedding_dims, label_smoothing).generator()

    dv_generator = Generator(x_dv_articles, x_dv_summaries,
                             document_max_sents, summary_max_sents,
                             document_max_words_per_sent, summary_max_words_per_sent,
                             vocabulary, pos_pairs, neg_pairs, embedding_dims, label_smoothing).generator()

    chkpath = path_models + "/" + name_models + "-{epoch:05d}-{val_loss:.5f}-{val_acc:.5f}.hdf5"

    checkpoint = ModelCheckpoint(chkpath, monitor=['val_loss', 'val_acc'],
                                 verbose=1, save_best_only=False,
                                 mode='min')
    callbacks = [checkpoint]
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
                                 pe_words = pe_words,
                                 dim_h = dim_h)

    ht.build()
    print(ht.model.summary())
    ht.compile(ht.model)



    ht.model.fit_generator(tr_generator, steps_per_epoch = steps_per_epoch,
                           epochs = epochs, validation_data = dv_generator,
                           validation_steps = validation_steps, verbose = 2,
                           callbacks = callbacks)

