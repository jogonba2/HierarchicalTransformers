from StringProcessing import StringProcessing

class Decoder:

    def __int__(self, attentions, sentence,
                document_max_sents,
                document_max_words_per_sent):

        self.attentions = attentions
        self.n_decoders = self.attentions.shape[0]
        self.n_heads = self.attentions.shape[1]
        self.n_sentences = self.attentions.shape[1]
        self.document_max_sents = document_max_sents
        self.document_max_words_per_sent = document_max_words_per_sent

    def _set_attentions(self, attentions):
        self.attentions = attentions

