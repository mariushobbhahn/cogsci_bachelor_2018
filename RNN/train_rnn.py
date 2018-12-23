from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape, Lambda, LocallyConnected1D, RepeatVector, Flatten, GaussianNoise
from data import character_trajectories
from keras.callbacks import ModelCheckpoint
from rnn import create_rnn
from rnn import train_rnn
import os
import numpy as np
import keras.backend as K

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
#RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_205_2_naked_v0.hdf5")
print(K.tensorflow_backend._get_available_gpus())

types = False
dim = 20
batch_size = 1
sparse = False
sparse_val = True
# load training and validation data  #x4_types_centers_only
(X_train, Y_train), (X_test, Y_test) = character_trajectories.load_data('../RNN/data/char_trajectories_{}_adverserial.pkl'.format(dim))
(X_val, Y_val), (_,_) = character_trajectories.load_data('../RNN/data/char_trajectories_{}x1_types_centers_only.pkl'.format(dim))

if sparse:
    """reduce the types to characters by taking the max along the axis"""
    print("test sample before the type reduction: ", X_train[0])
    X_train = np.amax(X_train, axis=3)
    X_test = np.amax(X_test, axis=3)
    print("test sample after type reduction: ", print(X_train[0]))

if sparse_val:
    #reshape validation set
    X_val = np.amax(X_val, axis=3)

print("x_train shape: " ,X_train.shape)
print("y_train shape: " ,Y_train.shape)
print("x_test shape: " ,X_test.shape)
print("y_test shape: " ,Y_test.shape)

"""reshape in and outputs"""

if types:
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
else:
    input_shape = (X_train.shape[1], X_train.shape[2])


def plot_sequence(sequence, swapaxis=False, deltas=False):
    if swapaxis:
        sequence = np.swapaxes(sequence,0 ,1)
    if deltas:
        x,y = np.cumsum(sequence[0]), np.cumsum(sequence[1])
    else:
        x,y = sequence[0], sequence[1]
    plt.plot(x, y)
    plt.show()


for i in [3]:
    RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_205_{}_adverserial_v{}.hdf5".format(dim, i))
    """set up model"""
    rnn = create_rnn(noise=True,
                     clip=True,
                     std_dev=0.3,
                     double_stacked=False,
                     hidden_size=205,
                     types=types,
                     input_shape=input_shape
                     )

    """train model"""
    train_rnn(model=rnn,
              x_train=X_train,
              y_train=Y_train,
              filename=RNN_FILE_MODEL,
              validation_split=0.0,
              validation_data = (X_val, Y_val),
              epochs=100,
              batch_size=batch_size)
