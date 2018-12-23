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
#from mvpa2.suite import *
from rnn import create_rnn

size = 128
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_seq_gen_std_{}_2.hdf5".format(size))

"""reshape in and outputs"""

(X_train, Y_train), (X_test, Y_test) = character_trajectories.load_data('../RNN/data/char_trajectories_2.pkl')
print("x_train shape: " ,X_train.shape)
print("y_train shape: " ,Y_train.shape)
print("x_test shape: " ,X_test.shape)
print("y_test shape: " ,Y_test.shape)

"""load model"""
rnn = create_rnn(clip=True,
                 noise=True,
                 std_dev=0,
                 hidden_size=size,
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

def plot_all_partitions(partition = part):
    #iterate all vectors in partition
    for i, vec in enumerate(partition):
        print("vector: ", vec)
        input_vec = prep_input(vec)
        prediction = rnn.predict(input_vec)
        plot_sequence(prediction, title='prediction_vector_{}'.format(vec), filename='{}_'.format(i) + str(vec), save=True, plot=False, swapaxis=True)

plot_all_partitions(create_partitions())
print(part)


def construct_distance_array(target, len_axis=100, dtw=False):
    axis = np.arange(start=0, stop=1.0000001, step=1/len_axis)
    grid_2d = np.array(np.meshgrid(axis, axis)).T.reshape(len_axis+1, len_axis+1 ,2)
    distance_array = np.zeros((len_axis + 1, len_axis + 1))
    assert(len(distance_array) == len(grid_2d))
    for i in range(len_axis + 1):
        for j in range(len_axis + 1):
            vec = prep_input(grid_2d[i][j])
            prediction = rnn.predict(vec)
            prediction = np.reshape(prediction, newshape=(205, 2))
            target = np.reshape(target, newshape=(205, 2))
            #print("shape of prediction: ", np.shape(prediction))
            #print("shape of target: ", np.shape(target))
            if dtw:
                distance_array[i][j] , _ = fastdtw(prediction, target, dist=euclidean)
            else:
                distance_array[i][j] = np.sqrt(np.mean(np.square(prediction - target)))

    return(distance_array)

def plot_heat_map(arr, filename='', save=False):
    fig, axis = plt.subplots()
    heatmap = axis.pcolor(arr, cmap=plt.cm.Blues)
    axis.set_yticks(np.arange(start=0, stop=arr.shape[0], step=1))
    axis.set_xticks(np.arange(start=0, stop=arr.shape[1], step=1))
    plt.colorbar(heatmap)
    plt.title(filename)
    if save:
        plt.savefig('../RNN/heat_maps/{}.png'.format(filename))
    plt.show()



test_a = np.reshape(Y_test[3], (1, 205, 2))
test_b = np.reshape(Y_test[1], (1, 205, 2))
plot_sequence(test_a, title='test_a', filename='', save=False, plot=True, swapaxis=True)
plot_sequence(test_b, title='test_b', filename='', save=False, plot=True, swapaxis=True)

array_a = construct_distance_array(test_a, dtw=True, len_axis=10)
print(array_a)
plot_heat_map(array_a, save=True, filename='distance_to_a_10_{}'.format(size))


array_b = construct_distance_array(test_b, dtw=True, len_axis=10)
print(array_b)
plot_heat_map(array_b, save=True, filename='distance_to_b_10_{}'.format(size))

