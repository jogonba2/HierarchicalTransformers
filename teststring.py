from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary as bv

sentence_attention_dims = [8, 8]
n_word_heads = [3, 3]  # Todas las capas con el mismo número de cabezales (facilita el decoder)
n_sentence_heads = [3, 3]  # Todas las capas con el mismo número de cabezales (facilita el decoder)
train_path = "./mini_cnndm.csv"
dev_path = "./mini_cnndm.csv"
vocabulary_path = "./vocab.vb"
max_vocabulary = 90
document_max_sents = 30
summary_max_sents = 4
document_max_words_per_sent = 25
summary_max_words_per_sent = 15
embedding_dims = 16

# Primero de todo, construir 1 sola vez el vocabulario #
#bv.save_vocab(train_path, vocabulary_path)
#############################################

vocabulary = bv.load_vocab(vocabulary_path, max_vocabulary)


sp = StringProcessing(vocabulary)

r = sp.represent_documents(["feature . the main president . in united states .",
                           "feature . the main president . in united states .",
                           "feature . the main president . in united states .",
                           "feature . the main president . in united states .",
                           "feature . the main president . in united states ."], 2, 5, " ", " . ")
print(r)