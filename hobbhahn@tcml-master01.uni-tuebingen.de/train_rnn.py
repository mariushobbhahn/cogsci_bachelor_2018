from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape, Lambda, LocallyConnected1D, RepeatVector, Flatten, GaussianNoise
from data import character_trajectories
from keras.callbacks import ModelCheckpoint
from rnn import create_rnn
from rnn import train_rnn
import os
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_seq_gen_std_128_2_repeller.hdf5")

# load training and validation data
(X_train, Y_train), (X_test, Y_test) = character_trajectories.load_data('../RNN/data/char_trajectories_2_repeller.pkl')
print("x_train shape: " ,X_train.shape)
print("y_train shape: " ,Y_train.shape)
print("x_test shape: " ,X_test.shape)
print("y_test shape: " ,Y_test.shape)

"""reshape in and outputs"""

def plot_sequence(sequence, swapaxis=False, deltas=False):
    if swapaxis:
        sequence = np.swapaxes(sequence,0 ,1)
    if deltas:
        x,y = np.cumsum(sequence[0]), np.cumsum(sequence[1])
    else:
        x,y = sequence[0], sequence[1]
    plt.plot(x, y)
    plt.show()


"""set up model"""
rnn = create_rnn(noise=False,
                 clip=False,
                 hidden_size=128,
                 input_shape=(X_train.shape[1], X_train.shape[2]))

"""train model"""
train_rnn(model=rnn,
          x_train=X_train,
          y_train=Y_train,
          filename=RNN_FILE_MODEL,
          epochs=int(2000/X_train.shape[2]),
          batch_size=1)