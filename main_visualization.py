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
    max_vocabulary = 200000 #100000
    train_path = "./test.csv"
    dev_path = "./test.csv"
    vocabulary_path = "./vocab.vb"

    # Training Parameters #
    pos_pairs = 32
    neg_pairs = 32
    epochs = 15
    steps_per_epoch = 750
    validation_steps = 250

    # Hierarchical Transformer Parameters #
    embedding_dims = 128
    n_word_encoders = 2
    n_sentence_encoders = 2
    attention_word_dims = 64
    attention_sentence_dims = 64
    n_w_heads = 8
    n_s_heads = 8
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
                             vocabulary, pos_pairs, neg_pairs).generator()

    dv_generator = Generator(x_dv_articles, x_dv_summaries,
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



    #ht.model.fit_generator(tr_generator, steps_per_epoch = steps_per_epoch,
    #                       epochs = epochs, validation_data = dv_generator,
    #                       validation_steps = validation_steps)

    #ht.save(ht.model, "./boso_version_weights.h5")
    ht.load(ht.model, "./transformer_12822646488-00058-0.19244-0.92675.hdf5")
    #ht.load(ht.model, "./transformer_12811646411-00026-0.05650-0.98062.hdf5")
    import matplotlib.pyplot as plt
    x_article = np.array([sp.represent_document(x_tr_articles[1319],
                                      document_max_sents,
                                      document_max_words_per_sent)])

    attns = ht.attn_model.predict(x_article)[0]

    i = 0
    ls_article = x_tr_articles[1319].split(" . ")
    ls_summary = x_tr_summaries[1319].split(" . ")
    print("ARTICLE\n")
    for i in range(len(ls_article)):
        print(i, ls_article[i])

    print("\n\nSUMMARY\n")
    for i in range(len(ls_summary)):
        print(i, ls_summary[i])

    # LAST || FIRST ENCODER ATTNS #

    last_encoder_attns = attns[1]
    #last_encoder_attns = np.expand_dims(last_encoder_attns, 0)
    w = 10
    h = 10
    fig = plt.figure(figsize=(16, 16))
    columns = 4
    rows = 2
    for i in range(1, columns * rows + 1):
        ind_enc = i - 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(last_encoder_attns[ind_enc], interpolation='nearest')
        plt.colorbar()
    plt.show()

    plt.imshow(last_encoder_attns[0], interpolation='nearest')
    plt.colorbar()
    plt.show()

    plt.imshow(last_encoder_attns[0], interpolation='nearest')
    plt.colorbar()
    plt.show()

    v_sum = last_encoder_attns.sum(axis=0)
    v_sum = np.expand_dims(v_sum.sum(axis=0), axis=0)
    top_3_sents = np.argsort(v_sum[0])[::-1][:3]
    print("Top 3: ", top_3_sents)

    plt.imshow(v_sum, interpolation='nearest')
    plt.colorbar()
    plt.show()

    # AVERAGE ALL SENTENCE ENCODER ATTNS #
    """
    v_sum = attns.sum(axis=0)
    v_sum = v_sum.sum(axis=0)
    v_sum = np.expand_dims(v_sum.sum(axis=0), axis=0)
    top_3_sents = np.argsort(v_sum[0])[::-1][:3]
    print("Top 3: ", top_3_sents)
    i = 0
    ls_article = x_tr_articles[315].split(" . ")
    for i in range(len(ls_article)):
        print(i, ls_article[i])

    plt.imshow(v_sum, interpolation='nearest')
    plt.colorbar()
    plt.show()
    """
