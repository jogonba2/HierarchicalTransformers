from StringProcessing import StringProcessing
from Utils import Utils
#from Visualization import Visualization
import numpy as np


class FastDecoder:

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


    def _gen_inputs(self, x):
        rx = self.sp.represent_document(x, self.document_max_sents, self.document_max_words_per_sent)
        word_mask_article = (rx>0).astype("int")
        sent_mask_article = (word_mask_article.sum(axis=1)>0).astype("int")
        pos_encodings_word_article = Utils.build_pe_word_encodings(self.matrix_encodings_words_article, word_mask_article)
        pos_encodings_sent_article = Utils.build_pe_sent_encodings(self.matrix_encodings_sents_article, sent_mask_article)
        return rx, word_mask_article, sent_mask_article, pos_encodings_word_article, pos_encodings_sent_article

    def _get_sample_attn(self, attns):
        return attns.sum(axis = 0) / attns.shape[0]

    # Save file: GENERATED_SUMMARY \t REFERENCE_SUMMARY #
    def _decode_samples(self, x, y, topk,
                        output_file,
                        average_encoders = False,
                        selected_encoder = -1,
                        selected_head = None,
                        visualization = False):
        gen_summaries = []
        #with open(output_file, "w", encoding="utf8") as fw:
        batch_size = 256 #512
        batch_rx, batch_wma, batch_sma, batch_pew, batch_pes = [], [], [], [], []
        for i in range(0, len(x), batch_size):
            print(i)
            b_x = x[i:i+batch_size]
            b_y = y[i:i+batch_size]
            for j in range(len(b_x)):
                rx, word_mask_article, sent_mask_article, pos_encodings_word_article, pos_encodings_sent_article = self._gen_inputs(b_x[j])
                batch_rx.append(rx)
                batch_wma.append(word_mask_article)
                batch_sma.append(sent_mask_article)
                batch_pew.append(pos_encodings_word_article)
                batch_pes.append(pos_encodings_sent_article)

            attns = self.attn_model.predict([batch_rx, batch_wma, batch_sma, batch_pew, batch_pes], batch_size = batch_size)

            for j in range(len(b_x)):
                if len(attns[j].shape) == 4:
                    if average_encoders:
                        selected_enc_attns = attns[j].sum(axis=0) / attns[j].shape[0]

                    else:
                        if selected_head is not None:
                            selected_enc_attns = attns[j][selected_encoder][selected_head]
                        else:
                            selected_enc_attns = attns[j][selected_encoder]

                else:
                    selected_enc_attns = attns[j]

                if selected_head is None:
                    selected_enc_attns = self._get_sample_attn(selected_enc_attns)

                x_lines = b_x[j].split("\n")
                lx_lines = len(x_lines)
                sent_pad_required = max(0, self.document_max_sents - lx_lines)
                attn = selected_enc_attns[sent_pad_required:, sent_pad_required:]
                sentence_attn = attn.sum(axis=0) / attn.shape[0]
                topk_sentences = sorted(np.argsort(sentence_attn)[::-1][:topk])
                summary = [x_lines[k] for k in topk_sentences]
                summary = "\n".join(summary)
                gen_summaries.append(summary)
                #fw.write(summary + "\t" + b_y[j] + "\n")

            batch_rx, batch_wma, batch_sma, batch_pew, batch_pes = [], [], [], [], []
        return gen_summaries
