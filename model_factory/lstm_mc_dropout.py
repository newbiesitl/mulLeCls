'''
   Created by Yubo Zhou on 28/03/19
'''

import keras


from keras.losses import binary_crossentropy

class SeqCLS(object):
    def __init__(self):
        self.m = None


    def configure(self, input_dim, seq_len, output_dim, h_dim, dropout,
                  loss='binary_crossentropy',
                  ):
        m = keras.models.Sequential()
        lstm_layer = keras.layers.LSTM(
            input_shape=(seq_len, input_dim),
            return_sequences=False,
            units=h_dim,
            dropout=dropout, recurrent_dropout=dropout,
        )
        m.add(lstm_layer)
        dense_h = keras.layers.Dense(
            units=h_dim,
            activation='selu',
        )
        m.add(dense_h); m.add(keras.layers.AlphaDropout(0.5))
        m.add(
            keras.layers.Dense(
                units=output_dim,
                activation='sigmoid',

            )
        )
        m.compile(loss=loss, optimizer='adam')
        self.m = m

    def fit(self, X, Y, epochs=50, batch_size=32, ):
        self.m.fit(X, Y, epochs=epochs, batch_size=batch_size)