from WordEncoderBlock import WordEncoderBlock
from SentenceEncoderBlock import SentenceEncoderBlock
from PositionalEncoding import PositionalEncoding
from LayerNormalization import LayerNormalization
from keras.models import Model
from keras.layers import (Input, GlobalMaxPooling1D, Dense,
                          SpatialDropout1D, Embedding,
                          TimeDistributed)

class HierarchicalTransformer():

    def __init__(self, max_vocabulary = 1000,
                 document_max_sents = 30,
                 summary_max_sents = 4,
                 max_words_per_sent = 25,
                 embedding_dims = 300,
                 output_word_encoder_dims = [300, 300, 300],
                 output_sentence_encoder_dims = [300, 300, 300],
                 word_attention_dims = [64, 64, 64],
                 sentence_attention_dims = [64, 64, 64],
                 n_word_heads = [8, 8, 8],
                 n_sentence_heads = [8, 8, 8]):

        self.document_max_sents = document_max_sents
        self.summary_max_sents = summary_max_sents
        self.max_words_per_sent = max_words_per_sent
        self.max_vocabulary = max_vocabulary
        self.embedding_dims = embedding_dims
        self.output_word_encoder_dims = output_word_encoder_dims
        self.output_sentence_encoder_dims = output_sentence_encoder_dims
        self.word_attention_dims = word_attention_dims
        self.sentence_attention_dims = sentence_attention_dims
        self.n_word_heads = n_word_heads
        self.n_sentence_heads = n_sentence_heads
        self.n_word_encoders = len(self.output_word_encoder_dims)
        self.n_sentence_encoders = len(self.output_sentence_encoder_dims)

    def build(self):

        self.input_article = Input(shape=(self.document_max_sents,
                                          self.max_words_per_sent))

        self.input_summary = Input(shape=(self.summary_max_sents,
                                          self.max_words_per_sent))

        self.embedding = Embedding(self.max_vocabulary, self.embedding_dims)

        # Get Word Embeddings (shared between branches) #
        self.e_article = self.embedding(self.input_article)
        self.e_summary = self.embedding(self.input_summary)

        # Adding Word embeddings and Positional embeddings #
        self.ep_article = TimeDistributed(PositionalEncoding())(self.e_article)
        self.ep_summary = TimeDistributed(PositionalEncoding())(self.e_summary)

        # Word Encoders (shared) #
        self.word_encoder_1 = WordEncoderBlock(self.output_word_encoder_dims[0],
                                               self.word_attention_dims[0],
                                               self.n_word_heads[0])

        z_article_word_encoder_1 = TimeDistributed(self.word_encoder_1)(self.ep_article)
        z_summary_word_encoder_1 = TimeDistributed(self.word_encoder_1)(self.ep_summary)
        z_article_word_encoder_1 = TimeDistributed(GlobalMaxPooling1D())(z_article_word_encoder_1)
        z_summary_word_encoder_1 = TimeDistributed(GlobalMaxPooling1D())(z_summary_word_encoder_1)

        # Sentence Encoders (shared) #
        self.sentence_encoder_1 = SentenceEncoderBlock(self.output_sentence_encoder_dims[0],
                                                       self.sentence_attention_dims[0],
                                                       self.n_sentence_heads[0])

        z_article_sentence_encoder_1,\
        attn_article_sentence_encoder_1 = self.sentence_encoder_1(z_article_word_encoder_1)

        #z_summary_sentence_encoder_1 = self.sentence_encoder_1(z_summary_word_encoder_1)

        self.model = Model(inputs = [self.input_article,
                                     self.input_summary],
                           outputs = [z_article_sentence_encoder_1])


    def save(self):
        pass

    def __str__(self):
        pass