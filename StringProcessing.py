import pandas as pd

class StringProcessing:

    def __init__(self, max_vocabulary,
                 document_max_sents,
                 document_max_words_per_sent,
                 summary_max_sents,
                 summary_max_words_per_sent,
                 vocabulary_path,
                 train_path,
                 dev_path = None,
                 test_path = None):

        self.train_path = train_path
        self.train = self.__load_csv_samples(self.train_path) if self.train_path else None
        self.dev_path = dev_path
        self.dev = self.__load_csv_samples(self.dev_path) if self.dev_path else None
        self.test_path = test_path
        self.test = self.__load_csv_samples(self.test_path) if self.test_path else None
        self.max_vocabulary = max_vocabulary
        self.document_max_sents = document_max_sents
        self.document_max_words_per_sent = document_max_words_per_sent
        self.summary_max_sents = summary_max_sents
        self.summary_max_words_per_sent = summary_max_words_per_sent
        self.vocabulary_path = vocabulary_path
        self.tokenizer_dict = {}
        self.untokenizer_dict = {}

    def __load_csv_samples(self, csv_path):
        aux_samples = []
        for chunk in pd.read_csv(csv_path, sep='\s*\t\s*', lineterminator="\n", chunksize=20000, engine="python", encoding="utf8"):
            aux_samples.append(chunk)

        csv_samples = pd.concat(aux_samples, axis=0)
        del aux_samples

        return csv_samples["TEXT"].tolist(), csv_samples["SUMMARY"].tolist()

    def load_vocabulary(self):
        pass

