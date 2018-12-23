from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "tutorial_RNN_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_tut_RNN.hdf5")


"""data generation"""

#generate input sequence (essentially just a seqence of 200 zeros with a 1 in the beginning)


NUM_SAMPLES = 200
time_steps = 360 #to have exactly one wave

input_seq = np.zeros(time_steps)
input_seq[0] = 1
inputs = np.tile(input_seq, (NUM_SAMPLES, 1))
inputs = np.reshape(inputs, (NUM_SAMPLES, time_steps, 1))

print("input_seq shape: ", np.shape(input_seq))
print("inputs shape: ", np.shape((inputs)))

#generate a sinosoidal with different, random starting points as a sequence:
def gen_sin_seq(fixed):
    #start with start_number
    if fixed:
        start_number = 1
    else:
        start_number = np.random.randint(0, 360)
    #create an array with length 200 representing the angles in  a circle
    seq = range(start_number, start_number + time_steps) #lets try 2.x waves
    #transform to radiant
    seq = np.array(seq) * (np.pi/180)
    #create the sequence of sin valuues
    sin_seq = np.sin(seq)
    return(sin_seq)

output_seq = gen_sin_seq(fixed=True)
outputs = np.tile(output_seq, (NUM_SAMPLES, 1))
outputs = np.reshape(outputs, (NUM_SAMPLES, time_steps, 1))
print("output_seq shape: ", np.shape(output_seq))
print("outputs: ", np.shape(outputs))
plt.plot(np.reshape(output_seq, (time_steps)))
plt.show()


"""set up model"""
hidden_size = 3
INPUT_SHAPE = (None, 1)

rnn = Sequential()
rnn.add(LSTM(hidden_size, return_sequences=True, input_shape=INPUT_SHAPE))
rnn.add(Dense(1, activation='linear'))

rnn.compile(loss='mean_squared_error', optimizer='adam')
rnn.summary()

#rnn.load_weights(RNN_FILE_MODEL)
"""train model"""
NUM_EPOCHS = 2000

checkpointer = ModelCheckpoint(filepath=RNN_FILE_MODEL, verbose=1, save_best_only=True)

rnn.fit(x=inputs,
        y=outputs,
        epochs=NUM_EPOCHS,
        batch_size=1,
        validation_split=0.2,
        callbacks=[checkpointer]
        )
