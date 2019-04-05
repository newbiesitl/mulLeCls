'''
   Created by Yubo Zhou on 28/03/19
'''

import keras
import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy
import numpy as np

class SeqCLS(object):
    def __init__(self):
        self.m = None
        self.model_t = None
        self.num_classes = 0

    def configure(self, input_dim, seq_len, output_dim, h_dim, dropout=0.5,
                  loss=binary_crossentropy, pretrained_embedding=None,
                  verbose=0,
                  ):
        self.num_classes = output_dim
        # with tf.device('/cpu:0'):
        if True:
            m = keras.models.Sequential()
            if pretrained_embedding is None:
                lstm_layer = keras.layers.LSTM(
                    input_shape=(seq_len, input_dim),
                    return_sequences=False,
                    units=h_dim,
                    dropout=dropout, recurrent_dropout=dropout,
                )
                m.add(lstm_layer)
                if verbose:
                    m.summary()
            else:
                m.add(pretrained_embedding)
                m.add(
                    keras.layers.LSTM(
                        return_sequences=False,
                        units=h_dim,
                        dropout=dropout, recurrent_dropout=dropout,
                    )
                )

        dense_h = keras.layers.Dense(
            units=h_dim,
            activation='selu',
        )
        m.add(dense_h); m.add(keras.layers.AlphaDropout(0.5))
        m.add(
            keras.layers.Dense(
                units=self.num_classes,
                activation='sigmoid',

            )
        )
        m.compile(loss=loss, optimizer='adam')
        self.m = m
        tensors = K.function([self.m.layers[0].input, K.learning_phase()],
                                          [self.m.layers[-1].output])
        self.model_t = tensors
        if verbose:
            self.m.summary()

    def fit(self, X, Y, epochs=50, batch_size=32, validation_split=.0, shuffle=True, verbose=2):
        self.m.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                   shuffle=shuffle, verbose=verbose)

        tensors = K.function([self.m.layers[0].input, K.learning_phase()],
                                          [self.m.layers[-1].output])
        self.model_t = tensors


    def predict_with_uncertainty(self, X, sim=1):
        result = self.sample_output(X, n_iter=sim)
        prediction = self.m.predict(X)
        result_cpy = np.swapaxes(result, 0, 1)
        result_cpy = np.swapaxes(result_cpy, 1, 2)
        uncertainties = np.zeros((result_cpy.shape[0], result_cpy.shape[1]))
        for data_idx in range(result_cpy.shape[0]):
            for topic_idx in range(result_cpy[data_idx].shape[0]):
                samples = result_cpy[data_idx][topic_idx]
                ret = np.histogram(samples, normed=True)
                bins, bin_edges = ret
                norm_bins = bins/np.sum(bins)
                for idx in range(len(norm_bins)):
                    if bin_edges[idx] < prediction[data_idx][topic_idx] <= bin_edges[idx+1]:
                        uncertainty_score = norm_bins[idx]
                uncertainties[data_idx][topic_idx] = uncertainty_score
        return prediction, uncertainties

    def sample_output(self, X, n_iter=1):
        result = np.zeros((n_iter,) + (X.shape[0], self.num_classes))
        for i in range(n_iter):
            result[i, :, :] = self.model_t((X, 1))[0]
        return result

    def summary(self):
        self.m.summary()
