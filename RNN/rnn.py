from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape, Lambda, LocallyConnected1D, RepeatVector, Flatten, GaussianNoise
from data import character_trajectories
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
#import matplotlib.pyplot as plt
import keras.backend as K

"""set up model"""
def create_rnn(input_shape,
              hidden_size,
              noise=False,
              std_dev=0.1,
              clip=False,
              double_stacked=False,
              types=False
              ):

    rnn = Sequential()
    if types:
        rnn.add(Reshape((input_shape[0], input_shape[1] * input_shape[2]), input_shape=input_shape))
    if noise:
        rnn.add(GaussianNoise(stddev=std_dev, input_shape=input_shape))
    if clip:
        rnn.add(Lambda(lambda x: K.clip(x, min_value=0, max_value=1)))
    rnn.add(LSTM(hidden_size, return_sequences=True, activation="tanh", stateful=False, input_shape=input_shape))
    if double_stacked:
        rnn.add(LSTM(hidden_size, return_sequences=True, activation="tanh", stateful=False))
    rnn.add(Dense(units=2, activation="linear"))

    """compile and show summary"""
    rnn.compile(loss='mean_squared_error', optimizer='adam')
    rnn.summary()
    return(rnn)


"""train model"""
def train_rnn(model,
              x_train,
              y_train,
              filename,
              epochs,
              validation_split=0.2,
              validation_data=None,
              batch_size=10
              ):

    checkpointer = ModelCheckpoint(filepath=filename, verbose=1, save_best_only=True)
    model.fit(x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split, #only takes the last x percent of your data. We present shuffled data.
            validation_data=validation_data,
            callbacks=[checkpointer],
            verbose=0
            )
