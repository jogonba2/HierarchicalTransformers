from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary
from HierarchicalTransformer import HierarchicalTransformer
from keras.utils.np_utils import to_categorical
import numpy as np

if __name__ == "__main__":

    document_max_sents = 30
    summary_max_sents = 4
    max_words_per_sent = 25 # Obligado compartido entre resumen y documentos por layer normalization
    embedding_dims = 64
    output_word_encoder_dims = [64, 64, 64] # Debe coincidir con los embeddings
    output_sentence_encoder_dims = [64, 64, 64]
    word_attention_dims = [16, 16, 16]
    sentence_attention_dims = [16, 16, 16]
    n_word_heads = [2, 2, 2]
    n_sentence_heads = [2, 2, 2]
    train_path = "./sample_set.csv"
    dev_path = "./sample_set.csv"
    max_vocabulary = 100

    # Build Vocabulary #
    #bv = BuildVocabulary()
    #string_processor = StringProcessing(train_path, dev_path)
    #print(string_processor.train)


    # Testing #
    max_vocabulary = 100
    n_samples = 2000

    x_pos_articles = np.random.randint(low = 1,
                                   high = 50,
                                   size = (n_samples, document_max_sents,
                                           max_words_per_sent)
                                   )

    x_pos_summaries = np.random.randint(low = 1,
                                   high = 50,
                                   size = (n_samples, summary_max_sents,
                                           max_words_per_sent)
                                   )
    x_neg_articles = np.random.randint(low = 51,
                                   high = 100,
                                   size = (n_samples, document_max_sents,
                                           max_words_per_sent)
                                   )

    x_neg_summaries = np.random.randint(low = 51,
                                   high = 100,
                                   size = (n_samples, summary_max_sents,
                                           max_words_per_sent)
                                   )

    x_articles = np.concatenate((x_pos_articles, x_neg_articles), axis=0)
    x_summaries = np.concatenate((x_pos_summaries, x_neg_summaries), axis=0)
    y = np.array([1 for i in range(n_samples)] + [0 for i in range(n_samples)])
    y = to_categorical(y, 2)
    print(x_articles.shape)
    print(x_summaries.shape)
    print(y.shape)

    ht = HierarchicalTransformer(max_vocabulary = max_vocabulary,
                                 document_max_sents = document_max_sents,
                                 summary_max_sents = summary_max_sents,
                                 max_words_per_sent = max_words_per_sent,
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
    #ht.compile(ht.attn_model)

    ht.model.fit([x_articles, x_summaries],
                  y = y, batch_size = 32,
                  epochs = 2, verbose = 1)

    #print(ht.model.predict([]))