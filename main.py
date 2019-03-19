from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary
from HierarchicalTransformer import HierarchicalTransformer
from keras.utils.np_utils import to_categorical
import numpy as np

if __name__ == "__main__":

    document_max_sents = 30
    summary_max_sents = 4
    max_words_per_sent = 25 # Obligado compartido entre resumen y documentos por layer normalization
    embedding_dims = 300
    output_word_encoder_dims = [300, 300, 300]
    output_sentence_encoder_dims = [300, 300, 300]
    word_attention_dims = [64, 64, 64]
    sentence_attention_dims = [64, 64, 64]
    n_word_heads = [8, 8, 8]
    n_sentence_heads = [8, 8, 8]
    train_path = "./sample_set.csv"
    dev_path = "./sample_set.csv"
    max_vocabulary = 1000

    # Build Vocabulary #
    #bv = BuildVocabulary()
    #string_processor = StringProcessing(train_path, dev_path)
    #print(string_processor.train)


    # Testing #
    max_vocabulary = 1000
    n_samples = 5000

    x_articles = np.random.randint(low = 1,
                                   high = max_vocabulary,
                                   size = (n_samples, document_max_sents,
                                           max_words_per_sent,
                                           1)
                                   )

    x_summaries = np.random.randint(low = 1,
                                   high = max_vocabulary,
                                   size = (n_samples, summary_max_sents,
                                           max_words_per_sent,
                                           1)
                                   )

    y = np.random.randint(0, 2, size=(n_samples))
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
    ht.model.summary()

    #print(ht.model.predict([]))