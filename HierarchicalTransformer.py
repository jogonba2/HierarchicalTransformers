from WordEncoderBlock import WordEncoderBlock
from SentenceEncoderBlock import SentenceEncoderBlock
from PositionalEncoding import PositionalEncoding
from MyMasking import MyMasking
from LayerNormalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.layers import (Input, GlobalMaxPooling1D, Dense,
                          Masking, Embedding, Flatten,
                          TimeDistributed, Lambda, Concatenate, SpatialDropout1D, Dropout)

class HierarchicalTransformer():

    def __init__(self, max_vocabulary = 1000,
                 document_max_sents = 30,
                 summary_max_sents = 4,
                 document_max_words_per_sent = 25,
                 summary_max_words_per_sent = 15,
                 embedding_dims = 300,
                 output_word_encoder_dims = [300, 300, 300],
                 output_sentence_encoder_dims = [300, 300, 300],
                 word_attention_dims = [64, 64, 64],
                 sentence_attention_dims = [64, 64, 64],
                 n_word_heads = [8, 8, 8],
                 n_sentence_heads = [8, 8, 8],
                 dropout_word_input = 0.,
                 dropout_sent_input = 0.,
                 dropout_word_output = 0.,
                 dropout_sent_output = 0.,
                 pe_sentences = True,
                 pe_words = True,
                 dim_h = 256):

        self.document_max_sents = document_max_sents
        self.summary_max_sents = summary_max_sents
        self.document_max_words_per_sent = document_max_words_per_sent
        self.summary_max_words_per_sent = summary_max_words_per_sent
        self.max_vocabulary = max_vocabulary
        self.embedding_dims = embedding_dims
        self.output_word_encoder_dims = output_word_encoder_dims
        self.output_sentence_encoder_dims = output_sentence_encoder_dims
        self.word_attention_dims = word_attention_dims
        self.sentence_attention_dims = sentence_attention_dims
        self.n_word_heads = n_word_heads
        self.n_sentence_heads = n_sentence_heads
        self.pe_sentences = pe_sentences
        self.pe_words = pe_words
        self.n_word_encoders = len(self.output_word_encoder_dims)
        self.n_sentence_encoders = len(self.output_sentence_encoder_dims)
        self.dropout_word_output = dropout_word_output
        self.dropout_sent_output = dropout_sent_output
        self.dropout_word_input = dropout_word_input
        self.dropout_sent_input = dropout_sent_input
        self.dim_h = dim_h

    def build(self):

        self.input_article = Input(shape=(self.document_max_sents,
                                          self.document_max_words_per_sent))

        self.input_summary = Input(shape=(self.summary_max_sents,
                                          self.summary_max_words_per_sent))

        self.mask_word_article = Input(shape=(self.document_max_sents,
                                              self.document_max_words_per_sent))

        self.mask_word_summary = Input(shape=(self.summary_max_sents,
                                              self.summary_max_words_per_sent))

        self.mask_sent_article = Input(shape=(self.document_max_sents,))
        self.mask_sent_summary = Input(shape=(self.summary_max_sents,))


        self.pos_encoding_word_article = Input(shape=(self.document_max_sents,
                                                      self.document_max_words_per_sent, self.embedding_dims))

        self.pos_encoding_word_summary = Input(shape=(self.summary_max_sents,
                                                      self.summary_max_words_per_sent, self.embedding_dims))

        self.pos_encoding_sent_article = Input(shape=(self.document_max_sents, self.embedding_dims))
        self.pos_encoding_sent_summary = Input(shape=(self.summary_max_sents, self.embedding_dims))


        self.embedding = Embedding(self.max_vocabulary, self.embedding_dims, mask_zero=False)

        # Get Word Embeddings (shared between branches) #
        self.e_article = self.embedding(self.input_article)
        self.e_summary = self.embedding(self.input_summary)

        # Masking de palabras #
        self.ep_article = TimeDistributed(MyMasking())(self.e_article, mask = self.mask_word_article)
        self.ep_summary = TimeDistributed(MyMasking())(self.e_summary, mask = self.mask_word_summary)

        # Adding Word embeddings and Positional embeddings #
        if self.pe_words:
            self.ep_article = TimeDistributed(PositionalEncoding())(self.ep_article, mask = self.pos_encoding_word_article)
            self.ep_summary = TimeDistributed(PositionalEncoding())(self.ep_summary, mask = self.pos_encoding_word_summary)

        # Dropout at input (word level)#
        #self.ep_article = TimeDistributed(SpatialDropout1D(self.dropout_word_input))(self.ep_article)
        #self.ep_summary = TimeDistributed(SpatialDropout1D(self.dropout_word_input))(self.ep_summary)

        # Word Encoders #

        ant_layers = (self.ep_article, self.ep_summary)
        for i in range(self.n_word_encoders):
            self.word_encoder = WordEncoderBlock(self.output_word_encoder_dims[i],
                                                 self.word_attention_dims[i],
                                                 self.n_word_heads[i], dropout = self.dropout_word_output)

            self.z_article_word_encoder = TimeDistributed(self.word_encoder)(ant_layers[0])
            self.z_summary_word_encoder = TimeDistributed(self.word_encoder)(ant_layers[1])

            self.z_article_word_encoder = TimeDistributed(MyMasking())(self.z_article_word_encoder, mask = self.mask_word_article) # Padding entre cada capa
            self.z_summary_word_encoder = TimeDistributed(MyMasking())(self.z_summary_word_encoder, mask = self.mask_word_summary)

            ant_layers = (self.z_article_word_encoder, self.z_summary_word_encoder)

        self.z_article_word_encoder = TimeDistributed(GlobalMaxPooling1D())(self.z_article_word_encoder)
        self.z_summary_word_encoder = TimeDistributed(GlobalMaxPooling1D())(self.z_summary_word_encoder)

        # Sentence Encoders #

        # Padding de frases #
        self.z_article_word_encoder = MyMasking()(self.z_article_word_encoder, mask = self.mask_sent_article)
        self.z_summary_word_encoder = MyMasking()(self.z_summary_word_encoder, mask = self.mask_sent_summary)

        # Positional Encodings para orden sobre frases #
        if self.pe_sentences:
            self.z_article_word_encoder = PositionalEncoding()(self.z_article_word_encoder, mask = self.pos_encoding_sent_article)
            self.z_summary_word_encoder = PositionalEncoding()(self.z_summary_word_encoder, mask = self.pos_encoding_sent_summary)

        # Dropout at input (sentence level)
        #self.z_article_word_encoder = SpatialDropout1D(self.dropout_sent_input)(self.z_article_word_encoder)
        #self.z_summary_word_encoder = SpatialDropout1D(self.dropout_sent_input)(self.z_summary_word_encoder)

        self.all_article_attns = []
        ant_layers = (self.z_article_word_encoder, self.z_summary_word_encoder)

        for i in range(self.n_sentence_encoders):
            self.sentence_encoder = SentenceEncoderBlock(self.output_sentence_encoder_dims[i],
                                                         self.sentence_attention_dims[i],
                                                         self.n_sentence_heads[i], dropout = self.dropout_sent_output)

            self.article_sentence_encoder = self.sentence_encoder(ant_layers[0])
            self.summary_sentence_encoder = self.sentence_encoder(ant_layers[1])

            self.z_article_sentence_encoder = Lambda(lambda x: x[0])(self.article_sentence_encoder)
            self.z_summary_sentence_encoder = Lambda(lambda x: x[0])(self.summary_sentence_encoder)

            self.attn_article_sentence_encoder = Lambda(lambda x: x[1])(self.article_sentence_encoder)
            self.all_article_attns.append(self.attn_article_sentence_encoder)

            # Masking entre cada capa #
            self.z_article_sentence_encoder = MyMasking()(self.z_article_sentence_encoder, mask = self.mask_sent_article)
            self.z_summary_sentence_encoder = MyMasking()(self.z_summary_sentence_encoder, mask = self.mask_sent_summary)

            ant_layers = (self.z_article_sentence_encoder, self.z_summary_sentence_encoder)

        # Prepare all attentions #
        if self.n_sentence_encoders > 1:
            self.all_article_attns = [Lambda(lambda a: K.expand_dims(a, 1))(x) for x in self.all_article_attns]
            self.all_article_attns = Concatenate(axis=1)(self.all_article_attns)

        ##########################


        self.z_article_sentence_encoder = GlobalMaxPooling1D()(self.z_article_sentence_encoder)
        self.z_summary_sentence_encoder = GlobalMaxPooling1D()(self.z_summary_sentence_encoder)

        self.difference = Lambda(lambda x: K.abs(x[0] - x[1]))([self.z_article_sentence_encoder,
                                                                self.z_summary_sentence_encoder])

        self.collapsed = Concatenate(axis=-1)([self.z_article_sentence_encoder,
                                               self.z_summary_sentence_encoder,
                                               self.difference])

        self.collapsed = LayerNormalization()(self.collapsed)

        self.h = Dense(self.dim_h, activation="relu")(self.collapsed)
        self.h = LayerNormalization()(self.h)
        self.output = Dense(2, activation="softmax")(self.h)

        self.model = Model(inputs = [self.input_article, self.mask_word_article, self.mask_sent_article, self.pos_encoding_word_article, self.pos_encoding_sent_article,
                                     self.input_summary, self.mask_word_summary, self.mask_sent_summary, self.pos_encoding_word_summary, self.pos_encoding_sent_summary],
                           outputs = [self.output])

        self.attn_model = Model(inputs = [self.input_article, self.mask_word_article, self.mask_sent_article, self.pos_encoding_word_article, self.pos_encoding_sent_article],
                                outputs = self.all_article_attns)

    def compile(self, model):
        model.compile(optimizer="adam", # adam
                      loss="categorical_crossentropy",
                      metrics = ["accuracy"])

    def save(self, model, f_name):
        model.save_weights(f_name)

    def load(self, model, f_name):
        model.load_weights(f_name)

    def __str__(self):
        pass
