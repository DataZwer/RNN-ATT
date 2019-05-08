import numpy as np


def load_word_embedding(word_index, file=None, trimmed_filename=None, load=False, dim=300):
    if load == True:  #
        with np.load(trimmed_filename) as data:
            return data["embeddings"]
    else:
        embeddings_index = {}
        with open(file, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Preparing embedding matrix.')
        embedding_matrix = np.zeros((len(word_index) + 1, dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        np.savez_compressed(trimmed_filename, embeddings=embedding_matrix)  #

    return embedding_matrix


'''
x_train, y_train, word_index, x_dev, y_dev = prepocess()

embedding_matrix = load_word_embedding(
    word_index,
    file=glove_embedding_path,
    trimmed_filename=glove_embedding_save,
    load=True,
    dim=300
)

'''



