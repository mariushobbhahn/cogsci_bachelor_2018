from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Flatten
from data import character_trajectories
from keras.callbacks import ModelCheckpoint
import os
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_v1.hdf5")
RNN_FILE_MODEL_weights = os.path.join(DIR_MODEL, "weights_RNN_v1.hdf5")

NUM_EPOCHS = 100
BATCH_SIZE = 1
hidden_size = 12 #mitjas paper suggests 12 - 60

# load training and validation data
(x_train, y_train), (x_test, y_test) = character_trajectories.load_data()
#print(np.shape(y_train))

x_train = np.reshape(x_train, (len(x_train), 20, 1))
y_train = np.reshape(y_train, (len(y_train), 2, 205))
x_test = np.reshape(x_test, (len(x_test), 20, 1))
y_test = np.reshape(y_test, (len(y_test), 2, 205))

# Get your input dimensions
# Input length is the length for one input sequence (i.e. the number of rows for your sample)
# Input dim is the number of dimensions in one input vector (i.e. number of input columns)
INPUT_LENGTH = y_train.shape[1]
INPUT_DIM = y_train.shape[2]
INPUT_SHAPE=(INPUT_LENGTH, INPUT_DIM)
# Output dimensions is the shape of a single output vector
# In this case it's just 1, but it could be more
output_dim = len(x_train[0])

rnn = Sequential()
#maybe add embedding layer?
rnn.add(LSTM(hidden_size, return_sequences=True, input_shape=INPUT_SHAPE))
rnn.add(LSTM(hidden_size, return_sequences=True))
rnn.add(Dense(output_dim=output_dim))



rnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
rnn.summary()

checkpointer = ModelCheckpoint(filepath=RNN_FILE_MODEL, verbose=1, save_best_only=True)
rnn.fit(x=y_train,
        y=x_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        #validation_data=(x_test, y_test),
        callbacks=[checkpointer]
        )