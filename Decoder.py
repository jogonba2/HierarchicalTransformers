from StringProcessing import StringProcessing
from Utils import Utils
from Visualization import Visualization
import numpy as np


class Decoder:

    def __init__(self, attn_model,
                document_max_sents,
                document_max_words_per_sent, sp, d_model):

        self.document_max_sents = document_max_sents
        self.document_max_words_per_sent = document_max_words_per_sent
        self.sp = sp
        self.d_model = d_model
        self.attn_model = attn_model
        self.matrix_encodings_words_article = Utils.precompute_word_pos_encodings(self.document_max_sents,
                                                                                  self.document_max_words_per_sent, self.d_model)
        self.matrix_encodings_sents_article = Utils.precompute_sent_pos_encodings(self.document_max_sents, self.d_model)

    def _get_sample_attns(self, x,
                          average_encoders = False,
                          selected_encoder = -1):

        rx = np.array([self.sp.represent_document(x, self.document_max_sents,
                                                  self.document_max_words_per_sent)])


        word_mask_article = (rx>0).astype("int")
        sent_mask_article = (word_mask_article.sum(axis=2)>0).astype("int")
        pos_encodings_word_article = np.array([Utils.build_pe_word_encodings(self.matrix_encodings_words_article, word_mask_article[0])])
        pos_encodings_sent_article = np.array([Utils.build_pe_sent_encodings(self.matrix_encodings_sents_article, sent_mask_article[0])])
        attns = self.attn_model.predict([rx, word_mask_article, sent_mask_article, pos_encodings_word_article, pos_encodings_sent_article])[0]

        # Tiene más de un encoder a nivel de frases
        # Por defecto, promedia los cabezales del último encoder
        # Se puede especificar si se quieren promediar todos los encoders o solo un encoder y cuál #
        if len(attns.shape) == 4:
            if average_encoders:
                avg_attns = attns.sum(axis=0) / attns.shape[0]
                return avg_attns, rx[0]
            else:
                selected_encoder_attns = attns[selected_encoder]
                return selected_encoder_attns, rx[0]

        # Tiene solo un encoder a nivel de frases
        else:
            return attns, rx[0]

    # Partiendo de _get_sample_attns que devuelve todos los cabezales, esta función es la estrategia para combinarlos,
    # implemento el promedio
    def _get_sample_attn(self, attns):
        return attns.sum(axis = 0) / attns.shape[0]

    def _decode_sample(self, x, topk,
                       average_encoders = False,
                       selected_encoder = -1,
                       visualization = False,
                       rows_visualization = None,
                       cols_visualization = None):

        attns, rx = self._get_sample_attns(x, average_encoders = average_encoders,
                                           selected_encoder = selected_encoder)

        if visualization: Visualization.visualize_attentions(attns, 16, 16, rows_visualization, cols_visualization)
        attn = self._get_sample_attn(attns)
        x_lines = x.split(" . ")
        lx_lines = len(x_lines)
        sent_pad_required = max(0, self.document_max_sents - lx_lines)
        if visualization: Visualization.visualize_attentions(attn, 16, 16, 1, 1)
        attn = attn[sent_pad_required:, sent_pad_required:]
        if visualization: Visualization.visualize_attentions(attn, 16, 16, 1, 1)
        sentence_attn = attn.sum(axis = 0) / attn.shape[0]
        topk_sentences = sorted(np.argsort(sentence_attn)[::-1][:topk])
        if visualization: Visualization.visualize_attentions(sentence_attn, 16, 16, 1, 1)
        summary = [x_lines[i] for i in topk_sentences]
        return " . ".join(summary)


    # Save file: GENERATED_SUMMARY \t REFERENCE_SUMMARY #
    def _decode_samples(self, x, y, topk,
                        output_file,
                        average_encoders = False,
                        selected_encoder = -1,
                        visualization = False,
                        rows_visualization = None,
                        cols_visualization = None):

        with open(output_file, "w", encoding="utf8") as fw:
            for i in range(len(x)):
                if i % 50 == 0: print(i)
                summary = self._decode_sample(x[i], topk,
                                              average_encoders = average_encoders,
                                              selected_encoder = selected_encoder,
                                              visualization = visualization,
                                              rows_visualization = rows_visualization,
                                              cols_visualization = cols_visualization)

                fw.write(summary + "\t" + y[i] + "\n")
