from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from keras.layers import Embedding
from config import model_folder
import os

global model_folder


def keyed_vector_to_keras_embedding(file_name):
    # embedding_file_name = 'google.google_keyed_vector_format.bin'
    kv = w2v_load_from_keyedvectors(file_name)
    return kv.get_keras_embedding()

def w2v_to_keyed_vector(input_file_name, output_file_name, binary=False):
    model_path = os.path.join(model_folder, input_file_name)
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=binary)
    word_vectors.save(os.path.join(model_folder, output_file_name))
    return

def w2v_load_from_w2v_format(file_name, binary=False):
    model_path = os.path.join(model_folder, file_name)
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=binary)
    return word_vectors


def w2v_load_from_keyedvectors(file_name):
    model_path = os.path.join(model_folder, file_name)
    word_vectors = KeyedVectors.load(model_path)
    return word_vectors



def build_embedding_layer(w2v_embedding, word_idx, vocab_size, max_sent_length, trainable=True):
    embedding_matrix = np.random.uniform(-0.454, 0.454, (vocab_size, w2v_embedding.vector_size))
    for idx, word in enumerate(word_idx):
        if word in w2v_embedding:
            embedding_matrix[idx] = w2v_embedding[word]
    return Embedding(vocab_size, w2v_embedding.vector_size,
                     weights=[embedding_matrix],
                     input_length=max_sent_length,
                     trainable=trainable,
                     input_shape=(max_sent_length,)
                     )


if __name__ == "__main__":
    '''
    Use as init script:
    load google.bin format and generate keyed vector format for faster loading
    '''
    w2v_file_name = 'GoogleNews-vectors-negative300.bin'
    keyed_vec_file_name = 'google_keyed_vector_format.bin'
    from config import model_folder
    w2v_to_keyed_vector(os.path.join(model_folder, w2v_file_name), os.path.join(model_folder, keyed_vec_file_name),
                        binary=True)
    embedding = w2v_load_from_keyedvectors(keyed_vec_file_name)
    keras_embedding = embedding.get_keras_embedding()
    i1 = input('type word1:\n')
    i2 = input('type word2:\n')
    v1 = embedding[i1]
    v2 = embedding[i2]
    ret = embedding.most_similar([v1, v2])
    while True:
        try:
            print(ret)
        except Exception as e:
            print(e)
            continue

