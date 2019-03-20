from StringProcessing import StringProcessing
from BuildVocabulary import BuildVocabulary
from HierarchicalTransformer import HierarchicalTransformer
from keras.utils.np_utils import to_categorical
import numpy as np

if __name__ == "__main__":

    document_max_sents = 30
    summary_max_sents = 4
    document_max_words_per_sent = 25
    summary_max_words_per_sent = 15
    embedding_dims = 16
    output_word_encoder_dims = [16, 16] # Debe coincidir con los embeddings para las conexiones residuales
    output_sentence_encoder_dims = [16, 16] # Debe coincidir con los embeddings para las conexiones residuales
    word_attention_dims = [4, 4]
    sentence_attention_dims = [8, 8]
    n_word_heads = [3, 3] # Todas las capas con el mismo número de cabezales (facilita el decoder)
    n_sentence_heads = [3, 3] # Todas las capas con el mismo número de cabezales (facilita el decoder)
    train_path = "./sample_set.csv"
    dev_path = "./sample_set.csv"
    max_vocabulary = 90

    # Build Vocabulary #
    #bv = BuildVocabulary()
    #string_processor = StringProcessing(train_path, dev_path)
    #print(string_processor.train)

    # Testing #

    max_vocabulary = 90
    n_samples = 10000

    x_pos_articles = np.random.randint(low = 1,
                                   high = 50,
                                   size = (n_samples, document_max_sents,
                                           document_max_words_per_sent)
                                   )

    x_pos_summaries = np.random.randint(low = 1,
                                   high = 50,
                                   size = (n_samples, summary_max_sents,
                                           summary_max_words_per_sent)
                                   )
    x_neg_articles = np.random.randint(low = 25,
                                   high = 90,
                                   size = (n_samples, document_max_sents,
                                           document_max_words_per_sent)
                                   )

    x_neg_summaries = np.random.randint(low = 25,
                                   high = 90,
                                   size = (n_samples, summary_max_sents,
                                           summary_max_words_per_sent)
                                   )

    x_articles = np.concatenate((x_pos_articles, x_neg_articles), axis=0)
    x_summaries = np.concatenate((x_pos_summaries, x_neg_summaries), axis=0)
    y = np.array([1 for i in range(n_samples)] + [0 for i in range(n_samples)])
    y = to_categorical(y, 2)

    # Training #
    ht = HierarchicalTransformer(max_vocabulary = max_vocabulary,
                                 document_max_sents = document_max_sents,
                                 summary_max_sents = summary_max_sents,
                                 document_max_words_per_sent = document_max_words_per_sent,
                                 summary_max_words_per_sent = summary_max_words_per_sent,
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

    #ht.load(ht.model, "./second_version_weights.h5")
    #ht.model.fit([x_articles, x_summaries],
    #              y = y, batch_size = 64,
    #              epochs = 1, verbose = 1)

    #ht.save(ht.model, "./second_version_weights.h5")
    #print(ht.model.predict([]))
    ht.load(ht.model, "./second_version_weights.h5")

    # Visualize Attention #

    #x_article = np.random.randint(low=25, high=90,
    #                              size = (1, document_max_sents,
    #                                      document_max_words_per_sent))
    #np.save("negative_sample.npy", x_article)


    # Con la NEGATIVA #

    x_article = np.load("negative_sample.npy")
    attns = ht.attn_model.predict(x_article)[0]
    print(attns.shape)
    exit()
    attn_head_0 = attns[0]
    attn_head_1 = attns[1]
    attn_head_2 = attns[2]

    import matplotlib.pyplot as plt

    print(sum(attn_head_0[0])) # = 1
    plt.imshow(attn_head_0, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    # Mostrar la más clara y la más oscura, a ver qué pasa ahí
    print("MAS CLARA:", attn_head_0[:, 13], "\n")
    print("\n\n\n MAS OSCURA:", attn_head_0[:, 4], "\n")

    print("Palabras de la frase MAS CLARA:", x_article[0][13], "\n")
    print("\n\n\nPalabras de la frase MAS OSCURA:", x_article[0][4], "\n")

    #print(attn_head_0.shape)
    #print(attn_head_1.shape)

    # ¿Cual es la frase que más palabras entre 40 y 50 tiene?,
    # la que más tenga debe ser la que menos atención debe tener
    # porque el articulo de ejemplo generado pertenece a la clase negativa
    # (entre 40 y 90), y lo que le distingue de la clase positiva son los valores
    # que no están en el solapamiento con esta (entre 40 y 50)
    # La frase 4 y la 19 son las que menos atención tienen (las que menos aportan i.e. muchos valores entre 40 y 50).
    # Una de las dos debe ser la que más valores en el solapamiento tiene, según las atenciones (LO ES!)#
    # Lo mismo pero para las frases con valores mayores de 50, estas son las que más atención deben tener,
    # porque son las que más contribuyen a decir la clase negativa #
    # según las atenciones, las que más valores mayores de 50 tienen son las frases 8 y 13 (LO SON!)#

    counts = []
    for i in range(len(x_article[0])):
        counts.append(0)
        for j in range(len(x_article[0][i])):
            if 25 < x_article[0][i][j] < 50:
                counts[-1] += 1

    # Frase clavada! (4) #
    print("Frases con mayor solapamiento:" , [i for i,val in enumerate(counts) if val==max(counts)])

    # Frases clavadas! (8 y 13) #
    print("Frases con menor solapamiento:" , [i for i,val in enumerate(counts) if val==min(counts)])


    # Testing pero con promedio de atencion de todos los cabezales #

    print(attns.shape)
    v_sum = attns.sum(axis=0) # Suma de todos los cabezales

    plt.imshow(v_sum, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    v_sum = np.expand_dims(v_sum.sum(axis=0), axis=0) # Suma vertical, cada frase como la suma de las atenciones para todas las otras frases

    plt.imshow(v_sum, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    print("La frase que menos aporta (promedio de todos los cabezales del último multihead attention): %d" % v_sum[0].argmin())
    print("La frase que más aporta (promedio de todos los cabezales del último multihead attention): %d" % v_sum[0].argmax())

    for i in range(len(counts)):
        print(i," - ",counts[i])

    # CON LA POSITIVA #

    #x_article = np.random.randint(low=1, high=50,
    #                              size = (1, document_max_sents,
    #                                      document_max_words_per_sent))
    #np.save("positive_sample.npy", x_article)

    x_article = np.load("positive_sample.npy")
    attns = ht.attn_model.predict(x_article)[0]  # (n cabezales ultimo encoder, n frases, n frases)
    attn_head_0 = attns[0]
    attn_head_1 = attns[1]
    attn_head_2 = attns[2]

    import matplotlib.pyplot as plt

    print(sum(attn_head_0[0]))  # = 1
    plt.imshow(attn_head_0, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    # Mostrar la más clara y la más oscura, a ver qué pasa ahí
    print("MAS CLARA:", attn_head_0[:, 13], "\n")
    print("\n\n\n MAS OSCURA:", attn_head_0[:, 4], "\n")

    print("Palabras de la frase MAS CLARA:", x_article[0][13], "\n")
    print("\n\n\nPalabras de la frase MAS OSCURA:", x_article[0][4], "\n")

    # print(attn_head_0.shape)
    # print(attn_head_1.shape)

    # ¿Cual es la frase que más palabras entre 40 y 50 tiene?,
    # la que más tenga debe ser la que menos atención debe tener
    # porque el articulo de ejemplo generado pertenece a la clase positiva
    # (entre 1 y 50), y lo que le distingue de la clase negativa son los valores
    # que no están en el solapamiento con esta (entre 40 y 50)
    # La frase y la  son las que menos atención tienen (las que menos aportan i.e. muchos valores entre 40 y 50).
    # Una de las dos debe ser la que más valores en el solapamiento tiene, según las atenciones (LO ES!)#
    # Lo mismo pero para las frases con valores menores de 40, estas son las que más atención deben tener,
    # porque son las que más contribuyen a decir la clase positiva #
    # según las atenciones, las que más valores menores de 40 tienen son las frases  y (LO SON!)#

    counts = []
    for i in range(len(x_article[0])):
        counts.append(0)
        for j in range(len(x_article[0][i])):
            if 25 < x_article[0][i][j] < 50:
                counts[-1] += 1

    # Frase clavada! () #
    print("Frases con mayor solapamiento:", [i for i, val in enumerate(counts) if val == max(counts)])

    # Frases clavadas! () #
    print("Frases con menor solapamiento:", [i for i, val in enumerate(counts) if val == min(counts)])

    # Testing pero con promedio de atencion de todos los cabezales #

    print(attns.shape)
    v_sum = attns.sum(axis=0)  # Suma de todos los cabezales

    plt.imshow(v_sum, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    v_sum = np.expand_dims(v_sum.sum(axis=0),
                           axis=0)  # Suma vertical, cada frase como la suma de las atenciones para todas las otras frases

    plt.imshow(v_sum, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    print("La frase que menos aporta (promedio de todos los cabezales del último multihead attention): %d" % v_sum[
        0].argmin())
    print("La frase que más aporta (promedio de todos los cabezales del último multihead attention): %d" % v_sum[
        0].argmax())

    for i in range(len(counts)):
        print(i," - ",counts[i])