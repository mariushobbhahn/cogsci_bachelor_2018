from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "tutorial_RNN_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_tut_RNN_v3.hdf5")

"""load model"""

rnn = load_model(RNN_FILE_MODEL)


"""make prediction"""
SEQ_LENGTH = 2000

input_seq = np.zeros(SEQ_LENGTH)
input_seq[0] = 1
input_seq = np.reshape(input_seq, newshape=(1, SEQ_LENGTH, 1))


prediction = rnn.predict(input_seq)
plt.plot(prediction[0])
plt.show()

"""test accuracy"""

