from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras import backend as K
from keras.losses import mean_squared_error
from keras.backend import get_value
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from data import character_trajectories
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from rnn import create_rnn
from minisom import MiniSom


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_205_4_noise_04_v0.hdf5")

"""reshape in and outputs"""

(X_train, Y_train), (X_test, Y_test) = character_trajectories.load_data('../RNN/data/char_trajectories_4.pkl')
print("x_train shape: " ,X_train.shape)
print("y_train shape: " ,Y_train.shape)
print("x_test shape: " ,X_test.shape)
print("y_test shape: " ,Y_test.shape)

"""load model"""
rnn = create_rnn(clip=True,
                 noise=True,
                 std_dev=0,
                 hidden_size=205,
                 double_stacked=False,
                 input_shape=(X_train.shape[1], X_train.shape[2]))
rnn.load_weights(RNN_FILE_MODEL)

def pad_class(class_vector, x=1):
    x_r = 205 - x #number of timesteps to fill
    class_vector = np.tile(class_vector, x)
    fill_vector = np.tile(np.zeros(X_train.shape[2]),(x_r,1))
    full_vector = np.vstack((class_vector, fill_vector))
    return(full_vector)

def prep_input(class_vector, x=1):
    class_vector = pad_class(class_vector, x)
    class_vector = np.reshape(class_vector, (1, 205, X_train.shape[2]))
    return(class_vector)

def plot_sequence(sequence, title, filename,swapaxis=False, deltas=False, save=False, plot=True):
    if swapaxis:
        sequence = np.swapaxes(sequence,1 ,2)
    if deltas:
        x,y = np.cumsum(sequence[0][0]), np.cumsum(sequence[0][1])
    else:
        x,y = sequence[0][0], sequence[0][1]
    plt.title(title)
    plt.plot(x, y)
    if save:
        plt.savefig('../RNN/vector_plots/{}.png'.format(filename))
        plt.clf()
    if plot:
        plt.show()

def partition(n, d, depth=0):
    if d == depth:
        return [[]]
    return [
        item + [i]
        for i in range(n+1)
        for item in partition(n-i, d, depth=depth+1)
        ]

def create_partitions(n = 10, d=X_train.shape[2]):
    # extend with n-sum(entries)
    n = 10
    d = X_train.shape[2]
    part = np.array([[n-sum(p)] + p for p in partition(n, d-1)])
    part = part/10
    return(part)

part = create_partitions()
print("len part: ", len(part))

def plot_all_partitions(partition = part):
    #iterate all vectors in partition
    for i, vec in enumerate(partition):
        print("i, vector: ", i, vec)
        input_vec = prep_input(vec)
        prediction = rnn.predict(input_vec)
        plot_sequence(prediction, title='{}_{}'.format(i, vec), filename='{}_{}'.format(i, vec), save=True, plot=False, swapaxis=True)

#plot_all_partitions()

#to create labels for our data, we look at the outcomes of test_rnn;
# if a letter cannot be clearly recognized it is labelled as x

labels_2d = np.array(['a', 'a', 'a', 'a', 'a', 'x', 'x', 'b', 'b', 'b', 'b'])

labels_4d_naked_v0 = np.array(['a', 'a', 'a', 'x', 'c', 'c', 'c', 'c', 'c', 'b', #0
                      'b', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c', #10
                      'b', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'b', #20
                      'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', #30
                      'c', 'c', 'c', 'c', 'c', 'a', 'x', 'c', 'c', 'c', #40
                      'c', 'a', 'c', 'c', 'c', 'c', 'a', 'c', 'c', 'c', #50
                      'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #60
                      'c', 'c', 'c', 'c', 'c', 'b', 'a', 'a', 'a', 'c', #70
                      'c', 'c', 'c', 'c', 'x', 'a', 'a', 'a', 'c', 'c', #80
                      'c', 'c', 'c', 'a', 'a', 'x', 'c', 'c', 'c', 'c', #90
                      'a', 'a', 'c', 'c', 'c', 'c', 'a', 'a', 'c', 'c', #100
                      'c', 'a', 'c', 'c', 'c', 'a', 'c', 'c', 'c', 'c', #110
                      'c', 'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', #120
                      'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'a', 'a', #130
                      'a', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'c', 'c', #140
                      'c', 'a', 'a', 'c', 'c', 'c', 'a', 'a', 'c', 'c', #150
                      'a', 'x', 'c', 'a', 'c', 'a', 'a', 'a', 'a', 'a', #160
                      'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', 'c', 'c', #170
                      'c', 'a', 'a', 'a', 'c', 'c', 'c', 'a', 'a', 'a', #180
                      'c', 'c', 'a', 'a', 'x', 'c', 'a', 'a', 'c', 'a', #190
                      'a', 'a', 'd', 'a', 'a', 'a', 'c', 'c', 'c', 'a', #200
                      'a', 'a', 'a', 'c', 'c', 'a', 'a', 'a', 'a', 'c', #210
                      'a', 'a', 'a', 'c', 'a', 'a', 'a', 'a', 'a', 'a', #220
                      'd', 'a', 'a', 'a', 'a', 'c', 'd', 'a', 'a', 'a', #230
                      'c', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', #240
                      'a', 'd', 'a', 'a', 'a', 'a', 'd', 'a', 'a', 'a', #250
                      'd', 'a', 'a', 'a', 'a', 'a', 'd', 'd', 'a', 'a', #260
                      'd', 'a', 'a', 'd', 'a', 'a', 'd', 'd', 'a', 'd',  #270
                      'd', 'd', 'd', 'd', 'd', 'd' #280 -286
                      ])


labels_4d_ds_v0 = np.array(['a', 'a', 'a', 'a', 'c', 'x', 'b', 'b', 'b', 'b', #0
                      'b', 'a', 'a', 'a', 'x', 'b', 'b', 'b', 'b', 'b', #10
                      'b', 'a', 'a', 'x', 'c', 'c', 'b', 'b', 'b', 'b', #20
                      'a', 'a', 'c', 'c', 'b', 'b', 'b', 'b', 'a', 'c', #30
                      'c', 'c', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'b', #40
                      'b', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #50
                      'c', 'b', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #60
                      'c', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', #70
                      'c', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'c', 'b', #80
                      'b', 'b', 'b', 'a', 'a', 'c', 'c', 'b', 'b', 'b', #90
                      'a', 'c', 'c', 'c', 'b', 'b', 'c', 'c', 'c', 'c', #100
                      'b', 'c', 'c', 'c', 'b', 'c', 'c', 'c', 'c', 'c', #110
                      'c', 'a', 'd', 'd', 'a', 'c', 'b', 'b', 'b', 'b', #120
                      'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'a', 'a', #130
                      'a', 'c', 'b', 'b', 'b', 'a', 'a', 'c', 'c', 'b', #140
                      'b', 'a', 'c', 'c', 'x', 'b', 'x', 'c', 'c', 'x', #150
                      'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'a', #160
                      'c', 'b', 'b', 'b', 'd', 'd', 'd', 'a', 'b', 'b', #170
                      'b', 'd', 'd', 'a', 'c', 'b', 'b', 'a', 'a', 'c', #180
                      'c', 'b', 'a', 'x', 'c', 'x', 'a', 'c', 'c', 'c', #190
                      'c', 'c', 'd', 'd', 'd', 'd', 'c', 'b', 'b', 'd', #200
                      'd', 'd', 'a', 'b', 'b', 'd', 'd', 'a', 'c', 'b', #210
                      'd', 'a', 'x', 'x', 'a', 'x', 'c', 'a', 'c', 'c', #220
                      'd', 'd', 'd', 'd', 'c', 'b', 'd', 'd', 'd', 'a', #230
                      'b', 'd', 'd', 'a', 'c', 'd', 'd', 'a', 'd', 'a', #240
                      'a', 'a', 'd', 'd', 'd', 'd', 'c', 'd', 'd', 'a', #250
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', #260
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd',  #270
                      'd', 'd', 'd', 'd', 'd', 'd' #280 -286
                      ])

labels_4d_noise_01_v0 = np.array(['a', 'a', 'a', 'a', 'a', 'x', 'c', 'b', 'b', 'b', #0
                      'b', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'b', 'b', #10
                      'b', 'a', 'a', 'a', 'a', 'x', 'c', 'x', 'b', 'b', #20
                      'a', 'a', 'a', 'x', 'c', 'c', 'b', 'b', 'a', 'a', #30
                      'a', 'c', 'c', 'c', 'b', 'a', 'x', 'c', 'c', 'c', #40
                      'x', 'x', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', #50
                      'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #60
                      'a', 'x', 'c', 'b', 'b', 'b', 'a', 'a', 'a', 'a', #70
                      'a', 'c', 'x', 'b', 'b', 'a', 'a', 'a', 'a', 'a', #80
                      'c', 'b', 'b', 'a', 'a', 'a', 'a', 'c', 'c', 'b', #90
                      'a', 'a', 'a', 'c', 'c', 'c', 'a', 'a', 'c', 'c', #100
                      'c', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c', 'c', #110
                      'c', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'b', 'b', #120
                      'a', 'a', 'a', 'a', 'a', 'c', 'b', 'b', 'a', 'a', #130
                      'a', 'a', 'x', 'c', 'b', 'a', 'a', 'a', 'a', 'c', #140
                      'c', 'a', 'a', 'a', 'c', 'c', 'a', 'a', 'x', 'c', #150
                      'x', 'x', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #160
                      'a', 'c', 'x', 'b', 'a', 'a', 'a', 'a', 'a', 'c', #170
                      'b', 'a', 'a', 'a', 'a', 'x', 'c', 'a', 'a', 'a', #180
                      'a', 'c', 'a', 'a', 'a', 'x', 'a', 'a', 'a', 'a', #190
                      'x', 'x', 'd', 'd', 'x', 'd', 'a', 'c', 'b', 'a', #200
                      'd', 'd', 'a', 'a', 'c', 'a', 'a', 'a', 'a', 'x', #210
                      'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', #220
                      'x', 'd', 'd', 'd', 'a', 'c', 'd', 'd', 'd', 'd', #230
                      'a', 'd', 'd', 'd', 'a', 'a', 'd', 'a', 'a', 'a', #240
                      'a', 'd', 'd', 'd', 'd', 'a', 'd', 'd', 'd', 'd', #250
                      'd', 'd', 'd', 'x', 'd', 'd', 'd', 'd', 'd', 'd', #260
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd',  #270
                      'd', 'd', 'd', 'd', 'd', 'd' #280 -286
                      ])



labels_4d_noise_02_v0 = np.array(['a', 'a', 'a', 'c', 'b', 'b', 'b', 'b', 'b', 'b', #0
                      'b', 'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'b', #10
                      'b', 'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'b', #20
                      'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'a', 'a', #30
                      'a', 'x', 'c', 'b', 'b', 'a', 'x', 'x', 'c', 'c', #40
                      'b', 'x', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', #50
                      'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #60
                      'c', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', #70
                      'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'b', #80
                      'b', 'b', 'b', 'a', 'a', 'a', 'x', 'b', 'b', 'b', #90
                      'a', 'a', 'a', 'c', 'b', 'b', 'a', 'a', 'c', 'c', #100
                      'b', 'x', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', #110
                      'c', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', #120
                      'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'a', 'a', #130
                      'a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'x', 'b', #140
                      'b', 'a', 'a', 'a', 'c', 'b', 'a', 'x', 'c', 'c', #150
                      'x', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #160
                      'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'b', 'b', #170
                      'b', 'a', 'a', 'a', 'a', 'b', 'b', 'a', 'a', 'a', #180
                      'x', 'b', 'a', 'a', 'a', 'c', 'a', 'a', 'x', 'c', #190
                      'c', 'c', 'd', 'd', 'd', 'd', 'c', 'b', 'b', 'a', #200
                      'd', 'd', 'a', 'c', 'b', 'a', 'a', 'a', 'a', 'b', #210
                      'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'x', #220
                      'd', 'd', 'd', 'd', 'a', 'b', 'd', 'd', 'd', 'd', #230
                      'a', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'a', 'a', #240
                      'a', 'd', 'd', 'd', 'd', 'a', 'd', 'd', 'd', 'd', #250
                      'd', 'd', 'd', 'x', 'd', 'd', 'd', 'd', 'd', 'd', #260
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd',  #270
                      'd', 'd', 'd', 'd', 'd', 'd' #280 -286
                      ])



labels_4d_noise_03_v0 = np.array(['a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c', 'b', #0
                      'b', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c', #10
                      'b', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'b', #20
                      'a', 'a', 'c', 'a', 'c', 'c', 'c', 'c', 'a', 'a', #30
                      'c', 'c', 'c', 'c', 'c', 'a', 'x', 'c', 'c', 'c', #40
                      'c', 'a', 'c', 'c', 'c', 'c', 'a', 'c', 'c', 'c', #50
                      'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #60
                      'c', 'c', 'c', 'c', 'c', 'b', 'a', 'a', 'a', 'c', #70
                      'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'c', 'c', #80
                      'c', 'c', 'c', 'a', 'a', 'x', 'c', 'c', 'c', 'c', #90
                      'a', 'a', 'c', 'c', 'c', 'c', 'a', 'a', 'c', 'c', #100
                      'c', 'a', 'c', 'c', 'c', 'a', 'c', 'c', 'c', 'c', #110
                      'c', 'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', #120
                      'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'a', 'a', #130
                      'a', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'c', 'c', #140
                      'c', 'a', 'a', 'c', 'c', 'c', 'a', 'a', 'c', 'c', #150
                      'a', 'x', 'c', 'a', 'c', 'a', 'a', 'a', 'a', 'a', #160
                      'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', 'c', 'c', #170
                      'c', 'a', 'a', 'a', 'c', 'c', 'c', 'a', 'a', 'a', #180
                      'c', 'c', 'a', 'a', 'a', 'c', 'a', 'a', 'c', 'a', #190
                      'a', 'a', 'd', 'a', 'a', 'a', 'c', 'c', 'c', 'a', #200
                      'a', 'a', 'a', 'c', 'c', 'a', 'a', 'a', 'a', 'c', #210
                      'a', 'a', 'a', 'c', 'a', 'a', 'a', 'a', 'a', 'a', #220
                      'd', 'a', 'a', 'a', 'a', 'c', 'd', 'a', 'a', 'a', #230
                      'c', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', #240
                      'a', 'd', 'a', 'a', 'a', 'a', 'd', 'a', 'a', 'a', #250
                      'd', 'a', 'a', 'a', 'a', 'a', 'd', 'd', 'a', 'a', #260
                      'd', 'a', 'a', 'a', 'd', 'a', 'd', 'd', 'a', 'd',  #270
                      'd', 'd', 'd', 'd', 'd', 'd' #280 -286
                      ])

labels_4d_noise_04_v0 = np.array(['a', 'a', 'a', 'a', 'a', 'x', 'b', 'b', 'b', 'b', #0
                      'b', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', #10
                      'b', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', #20
                      'a', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'a', 'a', #30
                      'a', 'c', 'x', 'b', 'b', 'a', 'a', 'c', 'c', 'x', #40
                      'b', 'a', 'c', 'c', 'c', 'x', 'c', 'c', 'c', 'c', #50
                      'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #60
                      'a', 'x', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', #70
                      'a', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'c', #80
                      'b', 'b', 'b', 'a', 'a', 'a', 'a', 'x', 'b', 'b', #90
                      'a', 'a', 'a', 'c', 'x', 'b', 'a', 'a', 'c', 'c', #100
                      'x', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', #110
                      'c', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', #120
                      'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', #130
                      'a', 'a', 'c', 'b', 'b', 'a', 'a', 'a', 'c', 'x', #140
                      'b', 'a', 'a', 'a', 'c', 'b', 'a', 'a', 'c', 'c', #150
                      'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', #160
                      'a', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'c', 'b', #170
                      'b', 'a', 'a', 'a', 'a', 'c', 'b', 'a', 'a', 'a', #180
                      'c', 'x', 'a', 'a', 'a', 'c', 'a', 'a', 'c', 'c', #190
                      'c', 'c', 'd', 'd', 'd', 'a', 'a', 'b', 'b', 'd', #200
                      'd', 'd', 'a', 'c', 'b', 'd', 'd', 'd', 'a', 'c', #210
                      'd', 'd', 'a', 'x', 'a', 'a', 'a', 'a', 'a', 'x', #220
                      'd', 'd', 'd', 'a', 'x', 'c', 'd', 'd', 'd', 'd', #230
                      'a', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', #240
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', #250
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', #260
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd',  #270
                      'd', 'd', 'd', 'd', 'd', 'd' #280 -286
                      ])


labels_4d_repeller_v0 = np.array(['a', 'a', 'a', 'x', 'x', 'x', 'b', 'b', 'b', 'b', #0
                      'b', 'a', 'a', 'a', 'x', 'x', 'b', 'b', 'b', 'b', #10
                      'b', 'a', 'a', 'a', 'x', 'x', 'b', 'b', 'b', 'b', #20
                      'a', 'a', 'x', 'x', 'b', 'b', 'b', 'b', 'a', 'a', #30
                      'x', 'x', 'x', 'b', 'b', 'a', 'x', 'x', 'x', 'b', #40
                      'b', 'c', 'c', 'c', 'x', 'b', 'c', 'c', 'c', 'x', #50
                      'c', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'x', 'x', #60
                      'x', 'x', 'b', 'b', 'b', 'b', 'a', 'a', 'x', 'x', #70
                      'x', 'x', 'b', 'b', 'b', 'a', 'a', 'x', 'x', 'x', #80
                      'x', 'b', 'b', 'a', 'a', 'x', 'x', 'x', 'x', 'b', #90
                      'a', 'a', 'x', 'x', 'x', 'b', 'a', 'a', 'x', 'b', #100
                      'b', 'a', 'c', 'c', 'b', 'c', 'c', 'c', 'c', 'c', #110
                      'c', 'a', 'c', 'x', 'x', 'x', 'x', 'x', 'b', 'b', #120
                      'a', 'x', 'x', 'x', 'x', 'x', 'x', 'b', 'a', 'a', #130
                      'x', 'x', 'x', 'x', 'x', 'a', 'a', 'x', 'x', 'x', #140
                      'x', 'a', 'a', 'x', 'x', 'x', 'a', 'a', 'x', 'x', #150
                      'a', 'x', 'x', 'c', 'c', 'c', 'a', 'x', 'x', 'x', #160
                      'x', 'x', 'x', 'x', 'a', 'x', 'x', 'x', 'x', 'x', #170
                      'x', 'a', 'x', 'x', 'x', 'x', 'x', 'a', 'x', 'x', #180
                      'x', 'x', 'a', 'x', 'x', 'x', 'a', 'x', 'x', 'a', #190
                      'x', 'x', 'x', 'x', 'd', 'd', 'd', 'x', 'x', 'x', #200
                      'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', #210
                      'x', 'x', 'x', 'x', 'a', 'x', 'x', 'a', 'x', 'a', #220
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', #230
                      'd', 'x', 'd', 'd', 'd', 'x', 'x', 'd', 'x', 'x', #240
                      'x', 'd', 'd', 'd', 'd', 'x', 'd', 'd', 'd', 'd', #250
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', #260
                      'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd',  #270
                      'd', 'd', 'd', 'd', 'd', 'd' #280 -286
                      ])

one_hot_4d = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

one_hot_2d =np.array([[1,0],
                      [0,1]])



def create_som(data,
               labels,
               one_hots,
               filename_load_weights,
               filename_save_weights,
               load_weights=False,
               num_iteration=100,
               plot_data=False,
               plot_labels=False,
               save_plot=False,
               plot_distance_map=False,
               show_activations=False,
               show_single_chars=False,
               filename_plots='unspecified.png'):
    assert len(data) == len(labels)

    size = int(np.ceil(np.sqrt(len(data))))
    input_len = len(data[0])
    # Initialization and training
    som = MiniSom(x=size, y=size, input_len=input_len, model=rnn, sigma=1.0, learning_rate=0.5)
    if load_weights:
        som.load_weights(filename=filename_load_weights)
    else:
        som.random_weights_init(data)
        print("Training...")
        som.train_random(data, num_iteration=num_iteration)  # random training
        print("\n...ready!")
        som.save_weights(filename=filename_save_weights)

    print("beginn mapping vectors")

    # Plotting the response for each pattern in the data set
    if plot_distance_map:
        plt.bone()
        plt.pcolor(som.distance_map().T)  # plotting the distance map as background
        plt.colorbar()
    else:
        plt.figure(figsize=(size, size))

    for i, data_point in enumerate(data):
        w = som.winner(data_point)  # getting the winner
        if plot_data:
            # place a string of the vector on the winning position for the sample
            plt.text(x=w[0],
                     y=w[1] + np.random.rand() * 0.9,
                     s=str(data_point),
                     size='small',
                     color='r'
                     )

        if plot_labels:
            #place the string of the label on the winning position for the sample
            plt.text(x=w[0] + 0.75,
                     y=w[1] + np.random.rand() * 0.9,
                     s=labels[i],
                     size='small',
                     color='b'
                     )

    #add axis
    plt.axis([0, size, 0, size])

    #save if specified
    if save_plot:
        plt.savefig('../RNN/SOM_graphics/{}.png'.format(filename_plots))
    plt.show()

    if show_activations:
        for i in range(len(one_hots)):
            plt.bone()
            plt.pcolor(som.activation_map(one_hots[i]))  # plotting the distance map as background
            plt.colorbar()
            plt.title('vec_{}'.format(one_hots[i]))
            plt.show()

    if show_single_chars:
        unique_labels = np.unique(labels)
        for unique_label in unique_labels:
            #plt.figure(figsize=(size, size))
            plt.bone()
            plt.pcolor(som.distance_map().T)  # plotting the distance map as background
            plt.colorbar()
            for i, data_point in enumerate(data):
                if unique_label == labels[i]:
                    w = som.winner(data_point)  # getting the winner
                    plt.text(x=w[0] + 0.75,
                             y=w[1] + np.random.rand() * 0.9,
                             s=labels[i],
                             size='small',
                             color='r'
                             )
                    #plot the vectors
                    plt.text(x=w[0],
                             y=w[1] + np.random.rand() * 0.9,
                             s=str(data_point),
                             size='small',
                             color='b'
                             )
            # add axis
            plt.axis([0, size, 0, size])
            plt.show()

create_som(part,
           labels_4d_noise_04_v0,
           one_hot_4d,
           load_weights=True, #if false this implies it is training the network from scratch
           filename_load_weights='../SOM_models/som_noise_04_17x17.pkl',
           filename_save_weights='../SOM_models/som_noise_04_17x17.pkl',
           plot_data=False,
           plot_labels=True,
           plot_distance_map=True,
           show_single_chars=True,
           show_activations=False)