from StringProcessing import StringProcessing
from collections import defaultdict

class BuildVocabulary:

    @staticmethod
    def build_vocab(train_file):
        text, summaries = StringProcessing.load_csv_samples(train_file)
        vocab = defaultdict(int)
        for i in range(len(text)):
            if not text[i] or not summaries[i]: continue
            t, s = text[i].split(), summaries[i].split()
            for w in t: vocab[w] += 1
            for w in s: vocab[w] += 1

        vocab = sorted(vocab.items(), key=lambda k_v: k_v[1], reverse=True)
        return vocab

    @staticmethod
    def save_vocab(train_file, dest_file):
        vocab = BuildVocabulary.build_vocab(train_file)
        with open(dest_file, "w", encoding="utf8") as fw:
            for (key, cnt) in vocab:
                fw.write(key + "\n")

    @staticmethod
    def load_vocab(vocab_file, max_vocab):
        vocab = {}
        with open(vocab_file, "r", encoding="utf8") as fr:
            # Desde 2 hasta |V| + 2, token 0 reservado para PAD y 1 para UNK
            for i in range(max_vocab):
                vocab[fr.readline().strip()] = i + 2
        return vocab

            