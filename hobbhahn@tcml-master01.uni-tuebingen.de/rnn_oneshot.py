from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras import backend as K
from keras.losses import mean_squared_error
from keras.backend import get_value
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from data import character_trajectories
import os
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt



DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_one_shot_std.hdf5")


# load training and validation data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = character_trajectories.load_data()
print("x_train/val/test shape: " ,x_train.shape)
print("y_train/val/test shape: " ,y_train.shape)

"""reshape data"""

x_train = np.reshape(x_train, (len(x_train), 1, 20))
y_train = np.reshape(y_train, (len(y_train), 205, 2))
x_test = np.reshape(x_test, (len(x_test), 1, 20))
y_test = np.reshape(y_test, (len(y_test), 2, 205))

"""load model"""

rnn = load_model(RNN_FILE_MODEL)


#"""
test_input_vector = np.zeros(20)
test_input_vector[0] = 1
test_input_vector[1] = 0
test_input_vector[2] = 0
test_input_vector[17] = 0

test_input_vector = np.reshape(test_input_vector, (1, 1, 20))

vector_1 = np.array([[[ 1.01932563,  0.0067674,   0.03216249, -0.10181677, 0.07051889,
    0.0119,      0.07255838,  0.02731965, -0.04472046, -0.06223465,
   -0.0584029,  -0.06077688, -0.07542864,  0.02443308,  0.04191943,
    0.06966833,  0.00218734,  0.01843605, -0.08308471,  0.09459123]]])

vector_2 = np.array([[[ 1.02074073e+00, -1.11184552e-02, -1.88182982e-02, -1.69649902e-02,
   -1.94152338e-02,  1.95134626e-03, -1.41121681e-02, -1.19832320e-02,
   -1.63884414e-02,-1.03982436e-02,  1.54452346e-02, -1.56152875e-02,
    1.77682707e-02,  1.12711159e-02,  7.18010650e-03,  1.77564639e-02,
   -8.13422713e-04,  2.12054073e-02, -1.15437376e-02,  4.40821842e-03]]])

vector_3 = np.array([[[ 0.96914922, -0.03073121,  0.00801072,  0.01304659,  0.0262378,
    0.03397391, -0.00122367, -0.00816541,  0.0045442,   0.00772148,
   -0.03294629, -0.01954611, -0.02256058, -0.00891733, -0.00704,
   -0.01949385, -0.009992,   -0.01830244, -0.01065982,  0.01506534]]])

vector_4 = np.array([[[ 0.98776148, -0.01967029,  0.01215587,  0.00993738, -0.02229045,
   -0.0012703,   0.01489168, -0.0145576,  -0.00704816,  0.01242052,
    0.01805613,  0.01276517,  0.00542985, -0.01464583,  0.01058811,
    0.00533549,  0.01818628,  0.00302563, -0.01866685, -0.00411082,]]])

#print("vector 3 before clipping: ", vector_3)
vector_2 = np.clip(vector_2, a_min= 0, a_max=1)

#print("vector 3 after clipping: ", vector_3)


prediction = rnn.predict(test_input_vector)
print("shape of prediction: ", np.shape(prediction))
#prediction = np.swapaxes(prediction, 1, 2)
print("shape of prediction: ", np.shape(prediction))
plt.plot(prediction[0][0], prediction[0][1])
plt.show()


#test zone for inverse classification:

output_tensor = rnn.output
test_output = y_test[0]
test_output = np.reshape(test_output, (1, 2, 205))
uniform_input = [1/20] * 20
uniform_input = np.reshape(uniform_input, (1,1,20))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
rnn.load_weights(RNN_FILE_MODEL)


#print("rnn.input: ", rnn.input)
#print("list of variable tensors first tensor: ", list_of_variable_tensors[0])
y_true = K.placeholder((1, 2, 205))
rnn_input = K.placeholder(shape=(1, 1, 20))
loss = K.mean(mean_squared_error(y_true=y_true, y_pred=rnn.output))
grads = K.gradients(loss = loss, variables=rnn.input)[0] #list_of_variable_tensors[0]
evaluate = K.function([rnn.input, y_true], [loss, grads])

random_training_example = np.random.random((1, 1, 20))

evaluated_gradients = sess.run(grads, feed_dict={rnn.input:test_input_vector, y_true:test_output})
print("gradients trial 100: ", evaluated_gradients)
updated_input = uniform_input + 0.0001 * evaluated_gradients
updated_input = np.clip(updated_input, 0,1)
print("updated input: ", updated_input)
evaluated_gradients = sess.run(grads, feed_dict={rnn.input:updated_input, y_true:test_output})
print("gradients trial 101: ", sess.run(grads, feed_dict={rnn.input:updated_input, y_true:test_output}))
print("loss and gradients 101: ", evaluate([updated_input, test_output]))
predicted_output = rnn.predict(updated_input)
print("manual loss:, ", sess.run(K.mean(mean_squared_error(y_true=test_output, y_pred=predicted_output))))
updated_input += 0.0001 * evaluated_gradients
updated_input = np.clip(updated_input, 0,1)
print("updated input 102: ", updated_input)


"""
evaluated_test_input = evaluate([test_input_vector, test_output])
evaluated_random_input = evaluate([random_training_example, test_output])
evaluated_uniform_input = evaluate([uniform_input, test_output])
print("evaluate uniform input: ", evaluated_uniform_input)
print("evaluate test letter input: ", evaluated_test_input)
print("evaluate random input: ",  evaluated_random_input)
#normalize input:
#evaluated_test_input = tf.nn.l2_normalize(evaluated_test_input[1])
#evaluated_test_input = np.reshape(evaluated_test_input[1], (1, -1))
#evaluated_test_input_1 = normalize(evaluated_test_input, axis=1)
#evaluated_uniform_input = tf.nn.l2_normalize(evaluated_uniform_input[1])
#evaluated_random_input = tf.nn.l2_normalize(evaluated_random_input[1])
print("evaluate test input normalized: ", evaluated_test_input)
#print("other norm: ", evaluated_test_input_1)
print("evaluate uniform input normalized: ", evaluated_uniform_input)
print("evaluate random input normalized: ", evaluated_random_input)

"""
#get_grads = K.function([rnn.input, y_true], K.gradients(loss, rnn.input))

#gradients = get_grads(uniform_tensor, test_output_tensor)


#gradients = K.gradients(output_tensor, list_of_variable_tensors)

#print("gradients: ", gradients)
"""

#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#evaluated_output_tensor = output_tensor.eval()
#print("manual loss: ", sess.run(manual_loss))


#print("normalized input: ", sess.run(evaluated_uniform_input))
#evaluated_test_input_sec_gen = evaluate([sess.run(evaluated_uniform_input), test_output])
#print("evaluated second iteration: ", evaluated_test_input_sec_gen)
#evaluated_test_input_sec_gen = tf.nn.l2_normalize(evaluated_test_input_sec_gen[1]
#print("normalized version: ", sess.run(evaluated_test_input_sec_gen))
#evaluated_test_input_sec_gen = tf.identity(evaluated_test_input_sec_gen[1])
#evaluated_test_input_third_gen = evaluate([sess.run(evaluated_test_input_sec_gen), test_output])
#print("evaluated third interation: ", evaluated_test_input_third_gen)
#gradients = K.gradients(output_tensor, list_of_variable_tensors)
#evaluated_gradients = sess.run(gradients, feed_dict={rnn.input:uniform_input})
#print(evaluated_gradients)
#print(evaluated_gradients[0])
#"""


# func = lambda r: K.mean(get_value(r.states[0]))
# print("func rnn: ", func(rnn.layers[3]))
#
# rnn.reset_states()
#
# print("func rnn: ",  func(rnn))

predicted_output = rnn.predict(test_input_vector)
error = K.mean(K.square(predicted_output - test_output), axis=-1)
print("error: ", sess.run(error))

#predicted_output = np.swapaxes(predicted_output, 1, 2)
print("shape of output: ", np.shape(predicted_output))
#plt.plot(predicted_output[0][0], predicted_output[0][1], 'o')
#plt.plot(predicted_output[0][0], predicted_output[0][1], 'o')
#plt.plot(predicted_output[0][0], predicted_output[0][1])
#plt.show()


def adam_optimizer(input_vector, gradient_vector, m_t1=0, v_t1=0, beta1=0.9, beta2=0.99, eta=0.001, epsilon=1e-8):
    m_t = m_t1 * beta1 + (1-beta1) * gradient_vector
    v_t = v_t1 * beta2 + (1-beta2) * (gradient_vector**2)

    epsilon = np.array([epsilon] * len(gradient_vector))
    dx = eta /(np.sqrt(v_t + epsilon)) * m_t
    output_vector = input_vector + dx

    return(output_vector, m_t, v_t)

rnn.load_weights(RNN_FILE_MODEL)

#iteratively generate gradients, send them back as input and see if they converge:
def iterate_gradients(x_vector, y_true):
    loss = 1000
    m_t1 = 0
    v_t1 = 0
    #while loss > 0.1:
    for i in range(100):
        [loss, grads] = evaluate([x_vector, y_true])
        x_vector, m_t1, v_t1 = adam_optimizer(input_vector=x_vector, gradient_vector=grads, m_t1=m_t1, v_t1=v_t1)
        #print("m_t1: ", m_t1, "v_t1: ", v_t1, "\n")
        #x_vector = np.clip(x_vector, a_min=0, a_max=1)
        #x_vector += 0.01 * grads
        print("loss: ", loss, "grads: ", grads, "new vector: ", x_vector, "sum grads: ", sess.run(tf.reduce_sum(grads)))

    x_vector = np.clip(x_vector, 0, 1)
    return(x_vector)

vector_5 = iterate_gradients(uniform_input, test_output)
vector_5 = np.reshape(vector_5, (1, 1, 20))

print("vector after gradient method: ", vector_5)
prediction = rnn.predict(vector_5)
#prediction = np.swapaxes(prediction, 1, 2)
plt.plot(prediction[0][0], prediction[0][1])
plt.show()