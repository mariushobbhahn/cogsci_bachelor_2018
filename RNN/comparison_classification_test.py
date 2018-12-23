from data import character_trajectories
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")

dim = 20
types = False
"""reshape in and outputs"""
(X_train, Y_train), (X_test, Y_test) = character_trajectories.load_data('../RNN/data/char_trajectories_{}.pkl'.format(dim))

"""remove the padding of the x_data"""
X_train = X_train[:,0,:]
X_test = X_test[:,0,:]

if types:
    """reduce the types to characters by taking the max along the axis"""
    print("test sample before the type reduction: ", X_test[0])
    X_train = np.amax(X_train, axis=2)
    X_test = np.amax(X_test, axis=2)
    print("test sample after type reduction: ", print(X_test[0]))


print("x_train shape: " ,X_train.shape)
print("y_train shape: " ,Y_train.shape)
print("x_test shape: " ,X_test.shape)
print("y_test shape: " ,Y_test.shape)


for i in range(5):
    """load model and weights"""
    print("model_{}".format(i))
    RNN_FILE_MODEL = os.path.join(DIR_MODEL, "comparison_model_{}_sparse1_v{}.hdf5".format(dim,i))
    rnn = load_model(RNN_FILE_MODEL)
    rnn.load_weights(RNN_FILE_MODEL)

    """predict results"""
    prediction = rnn.predict(Y_test)
    error = np.sqrt(np.mean(np.square(prediction - X_test)))
    prediction = np.around(prediction, decimals=0)
    accuracy = np.mean(np.logical_and.reduce(prediction == X_test, axis = -1))
    #np.equal(prediction, X_test, axis = 0)

    #print("prediction: ", prediction)
    print("accuracy: ", accuracy)
    print("error: ", error)
    """compare to real results"""
    #print("real: ", X_test)