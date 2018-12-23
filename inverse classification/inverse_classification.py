from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras import backend as K
from keras.losses import mean_squared_error
from keras.backend import get_value
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from RNN.data import character_trajectories
import os
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


dim = 20
version = 2
types = False
sparse = False
repeller = False
#DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = "../RNN/rnn_models"
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "model_RNN_205_{}_adverserial_v{}.hdf5".format(dim, version))

print(RNN_FILE_MODEL)

"""reshape in and outputs"""

(X_train, Y_train), (X_test, Y_test) = character_trajectories.load_data('../RNN/data/char_trajectories_{}.pkl'.format((dim)))

if(types or sparse):
    """reduce the types to characters by taking the max along the axis"""
    print("test sample before the type reduction: ", X_test[0])
    #x_train = np.amax(x_train, axis=1) train isnt used anyways
    X_test = np.amax(X_test, axis=3)
    print("test sample after type reduction: ", print(X_test[0]))

#save time
if dim == 20:
    Y_test = Y_test[0:250,:,:]

print("x_train shape: " ,X_train.shape)
print("y_train shape: " ,Y_train.shape)
print("x_test shape: " ,X_test.shape)
print("y_test shape: " ,Y_test.shape)

"""load model"""

rnn = load_model(RNN_FILE_MODEL)


#"""
def pad_class_types(class_array, number_of_paddings=204):
    shape = np.shape(class_array)
    #print("shape: ", shape)
    padding = np.array([np.zeros(shape=shape)] * number_of_paddings)
    #print("padding: ", np.shape(padding))
    class_array = np.expand_dims(class_array, axis=0)
    padded_array = np.vstack((class_array, padding))
    return(padded_array)

def pad_class(class_vector, x=1):
    x_r = 205 - x #number of timesteps to fill
    class_vector = np.tile(class_vector, x)
    fill_vector = np.tile(np.zeros(X_train.shape[2]),(x_r,1))
    full_vector = np.vstack((class_vector, fill_vector))
    return(full_vector)

def prep_input(class_vector, x=1, types=False):

    if types:
        class_vector = pad_class_types(class_vector)
        class_vector = np.reshape(class_vector, (1, 205, X_train.shape[2], X_train.shape[3]))
    else:
        class_vector = pad_class(class_vector, x)
        class_vector = np.reshape(class_vector, (1, 205, X_train.shape[2]))
    return(class_vector)

test_input_vector = np.zeros(X_train.shape[2])
test_input_vector[0] = 1 #a
test_input_vector[1] = 0 #b
"""
test_input_vector[2] = 0 #c
test_input_vector[3] = 0 #d
test_input_vector[4] = 0 #e
test_input_vector[5] = 0 #g
test_input_vector[6] = 0 #h
test_input_vector[7] = 0 #l
test_input_vector[8] = 0 #m
test_input_vector[9] = 0 #n
test_input_vector[10] = 0 #o
test_input_vector[11] = 0 #p
test_input_vector[12] = 0 #q
test_input_vector[13] = 0 #r
test_input_vector[14] = 0 #s
test_input_vector[15] = 0 #u
test_input_vector[16] = 0 #v
test_input_vector[17] = 0 #w
test_input_vector[18] = 0 #y
test_input_vector[19] = 0 #z
"""

if types:
    uniform_input = np.full((X_train.shape[2], X_train.shape[3]), 1/(X_train.shape[2] * X_train.shape[3]))
    uniform_input = prep_input(uniform_input, types=True)
else:
    uniform_input = [1/X_train.shape[2]] * X_train.shape[2]
    uniform_input = prep_input(uniform_input)

#print("uniform_input: ", uniform_input)

def plot_sequence(sequence, title, swapaxis=False, deltas=False):
    if swapaxis:
        sequence = np.swapaxes(sequence,1 ,2)
    if deltas:
        x,y = np.cumsum(sequence[0][0]), np.cumsum(sequence[0][1])
    else:
        x,y = sequence[0][0], sequence[0][1]
    plt.title(title)
    plt.plot(x, y)
    plt.show()

#prediction = rnn.predict(test_input_vector)
#plot_sequence(prediction, swapaxis=True, deltas=False, title='test_input_vector')


#test zone for inverse classification:

output_tensor = rnn.output
test_output = Y_test[2]
test_output = np.reshape(test_output, (1, 205, 2))
#plot_sequence(test_output, swapaxis=True, deltas=False, title='test_output')


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
rnn.load_weights(RNN_FILE_MODEL)


y_true = K.placeholder((1, 205, 2))
loss = mean_squared_error(y_true=y_true, y_pred=rnn.output)
grads = K.gradients(loss = loss, variables=rnn.input)[0]
evaluate = K.function([rnn.input, y_true], [loss, grads])


def adam_optimizer(input_vector, gradient_vector, m_t1=0, v_t1=0, beta1=0.9, beta2=0.99, eta=0.0001, epsilon=1e-8):
    #get moment terms
    m_t = m_t1 * beta1 + (1-beta1) * gradient_vector
    v_t = v_t1 * beta2 + (1-beta2) * np.square(gradient_vector)

    epsilon = np.full(shape=np.shape(gradient_vector), fill_value=epsilon)
    dx = (eta * m_t)/(np.sqrt(v_t) + epsilon)
    output_vector = input_vector - dx

    return(output_vector, m_t, v_t)


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

rnn.load_weights(RNN_FILE_MODEL)

#iteratively generate gradients, send them back as input and see if they converge:
def iterate_gradients(x_vector,
                      y_true,
                      iterations=200,
                      learning_rate=0.05,
                      #threshold=0.99,
                      adam=True,
                      softmax=False,
                      print_iterations=False,
                      plot_iterations=False,
                      plot_y_true=False
                      ):

    m_t1 = 0
    v_t1 = 0
    i = 0
    while(i < iterations): #and max(x_vector[0][0]/sum(x_vector[0][0])) <= threshold):
        i += 1
        [loss, grads] = evaluate([x_vector, y_true])
        grads = grads[0][0] #we only want the very first entries
        if plot_y_true:
            print("y_true shape: ", np.shape(y_true))
            y_true = np.swapaxes(y_true, 1, 2)
            plt.plot(y_true[0][0], y_true[0][1])
            y_true = np.swapaxes(y_true, 1, 2)
            plt.show()
        #add gradients:
        if adam:
            x_vector[0][0], m_t1, v_t1 = adam_optimizer(input_vector=x_vector[0][0], gradient_vector=grads, eta=learning_rate, m_t1=m_t1, v_t1=v_t1)
        else:
            x_vector[0][0] -= learning_rate * grads
        if softmax:
            ps = np.exp(x_vector[0][0])
            x_vector[0][0] = ps/sum(ps)
        #clip vector:
        x_vector = np.clip(x_vector, a_min=0, a_max=1)
        #normalize such that sum is 1:
        x_vector[0][0] = x_vector[0][0]/(np.sum(x_vector[0][0]) + 0.00000001) #to prevent NaN
        if print_iterations:
            print("loss: ", np.mean(loss),
                    "\ngrads: ", grads,
                    "\nnew vector: ", x_vector[0][0]
                  )
        if plot_iterations:
            prediction = rnn.predict(x_vector)
            prediction = np.swapaxes(prediction, 1, 2)
            plt.plot(prediction[0][0], prediction[0][1])
            plt.show()


    x_vector = np.clip(x_vector, 0, 1)
    return(x_vector)


"""
vector_5 = iterate_gradients(uniform_input, test_output, iterations=200, learning_rate=0.05, print_iterations=True, softmax=False )
print("vector 5 shape: ", np.shape(vector_5))
vector_5 = np.reshape(vector_5, (1, 205, X_train.shape[2]))

print("vector after gradient method: ", vector_5[0][0])
prediction = rnn.predict(vector_5)
prediction = np.swapaxes(prediction, 1, 2)
plot_sequence(prediction, swapaxis=False, deltas=False, title='prediction after gradient method')
"""

inp_list = []
id = np.identity(dim)
for i in range(len(id[0])):
    inp_list.append(prep_input(id[i]))

print("shape of list: ", np.shape(inp_list))



def evaluate_ic_quality(num_samples=Y_test.shape[0],
                        y_test=Y_test,
                        x_test=X_test,
                        inputs=inp_list,   #uniform_input
                        repeller=False,
                        types=False,
                        plot_targets=True,
                        plot_predictions=True,
                        print_correct=True,
                        print_predictions=True,
                        print_targets=True):
    errors = []
    correct = []
    for i in range(num_samples):
        target = y_test[i]
        target = np.reshape(target, (1, 205, 2))
        if plot_targets:
            plot_sequence(target, swapaxis=True, deltas=False, title='test_output_{}'.format(i))

        list_of_results = []
        print("length of inputs: ", len(inputs))
        for j in range(len(inputs)):
            print("current input: ", inputs[j][0][0])
            vector = iterate_gradients(inputs[j], target, iterations = 1, print_iterations=False, plot_iterations=False ,plot_y_true=False)
            vector = vector[0][0]
            if print_predictions:
                print("vector_{} after gradient iteration: ".format(i), vector)
                if types:
                    print("vector_{} after sum reduction over types dimension: ".format(i), np.sum(vector, axis=1))
            if print_targets:
                print("class_{}: ".format(i), x_test[i][0])
            #calculate error:
            if types:
                error = np.sqrt(np.mean(np.square(np.sum(vector, axis=1) - x_test[i][0])))
            else:
                error = np.sqrt(np.mean(np.square(vector - x_test[i][0])))
            print("error: ", error)
            errors.append(error)
            #check if classification is correct
            list_of_results.append(vector)

        print("list of results: ", list_of_results)
        vector = np.mode(list_of_results)
        #repeller:
        if repeller:
            n = len(vector)
            rounded_vec = np.around(vector * n)/n
            if not np.array_equal(rounded_vec, np.array(np.array([1/n] * n ))):
                rounded_vec = np.zeros_like(vector)
                rounded_vec[np.argmax(vector)] = 1
        elif types:
            #take the sum over the axis and reduce to dimension of 2:
            reduced_vector = np.sum(vector, axis=1)
            rounded_vec = np.zeros_like(reduced_vector)
            rounded_vec[np.argmax(reduced_vector)] = 1
            print("rounded reduced vec: ", rounded_vec)
        else:
            rounded_vec = np.zeros_like(vector)
            rounded_vec[np.argmax(vector)] = 1
        #compare result and target:
        if np.array_equal(rounded_vec, x_test[i][0]):
            correct.append(1)
        else:
            correct.append(0)
        if print_correct:
            print("correct: ", np.array_equal(rounded_vec, x_test[i][0]))
        #predict and plot
        if plot_predictions:
            prediction = rnn.predict(vector)
            plot_sequence(prediction, swapaxis=True, deltas=False, title='vector_{}_prediction'.format(i))
        #whether the prediction was correct

    accuracy = sum(correct)/num_samples
    mean_error = np.mean(errors)
    return(accuracy, mean_error)




print(evaluate_ic_quality(inputs=inp_list, plot_targets=False, plot_predictions=False, repeller=repeller, types=types))

"""
2_naked_v0: (0.6984126984126984, 0.3622880332147765)
2_naked_v1: (0.5714285714285714, 0.440150868229461)
2_naked_v2: (0.8095238095238095, 0.3225553659084296)
2_naked_v3: (0.8412698412698413, 0.1726171253061654)
2_naked_v4: (0.38095238095238093, 0.5996755467597744)


2_ds_v0: (0.5873015873015873, 0.48370579609339315)
2_ds_v1: (0.42857142857142855, 0.5764464341684693)
2_ds_v2: (0.3968253968253968, 0.4961492109753141)
2_ds_v3: (0.746031746031746, 0.3256718465647067)
2_ds_v4: (0.5238095238095238, 0.5576492647733844)


2_noise_01_v0: (0.9047619047619048, 0.1315291134566969)
2_noise_01_v1: (0.9206349206349206, 0.10450408448673962)
2_noise_01_v2: (0.6507936507936508, 0.41821481911922254)
2_noise_01_v3: (0.4603174603174603, 0.43645874488716335)
2_noise_01_v4: (0.4126984126984127, 0.41279800669115124)


2_noise_02_v0: (1.0, 0.0759593499013347)
2_noise_02_v1: (0.9682539682539683, 0.09276134912124992)
2_noise_02_v2: (1.0, 0.018963177939329164)
2_noise_02_v3: (0.9365079365079365, 0.15582147132465818)
2_noise_02_v4: (1.0, 0.07781698176242056)


2_noise_03_v0: (1.0, 0.04146389184518892)
2_noise_03_v1: (1.0, 0.09865228048290489)
2_noise_03_v2: (1.0, 0.12268841968810419)
2_noise_03_v3: (1.0, 0.06950291878573156)
2_noise_03_v4: (1.0, 0.09061335822971789)


2_noise_04_v0: (1.0, 0.0945932183214874)
2_noise_04_v1: (1.0, 0.10361514214837172)
2_noise_04_v2: (1.0, 0.1553425721267589)
2_noise_04_v3: (1.0, 0.138311098662929)
2_noise_04_v4: (1.0, 0.1151927357249817)


2_repeller_v0: (0.8829787234042553, 0.11834418158637727)
2_repeller_v1: (0.6914893617021277, 0.15382725841758407)
2_repeller_v2: (0.723404255319149, 0.17655094738258284)
2_repeller_v3: (0.8404255319148937, 0.15140733357219216)
2_repeller_v4: (0.8297872340425532, 0.14625343689692918)

2_types_v0: (0.5714285714285714, 0.46909085245645515)
2_types_v1: (0.6031746031746031, 0.4460858245199946)
2_types_v2: (0.9365079365079365, 0.15033957250802385)
2_types_v3: (1.0, 0.2812244623341086)
2_types_v4: (0.8412698412698413, 0.33130075138062076)


2_types_noise_03_v0: (1.0, 0.18220864906477605)
2_types_noise_03_v1: (0.9365079365079365, 0.2904404586609495)
2_types_noise_03_v2: (0.9523809523809523, 0.24518237105749247)
2_types_noise_03_v3: (0.3968253968253968, 0.543596142872431)
2_types_noise_03_v4: (0.9841269841269841, 0.32642376233545495)


2_sparse_noise_04_v0: (1.0, 0.16022956536268024)
2_sparse_noise_04_v1: (0.9365079365079365, 0.10993446524355528)
2_sparse_noise_04_v2: (1.0, 0.0946226152179276)
2_sparse_noise_04_v3: (1.0, 0.15347101859370987)
2_sparse_noise_04_v4: (0.9365079365079365, 0.14455421041336294)

2_sparse1_noise_04_v0: (1.0, 0.15154102144854614)
2_sparse1_noise_04_v1: (1.0, 0.13734922995186508)
2_sparse1_noise_04_v2: (1.0, 0.10201477310041672)
2_sparse1_noise_04_v3: (1.0, 0.16018938842582137)
2_sparse1_noise_04_v4: (1.0, 0.11483627603459963)


2_adverserial_v0: (1.0, 0.06338422275938545)
2_adverserial_v1: (1.0, 0.05576379135488315)
2_adverserial_v2: (1.0, 0.05642754691505229)
2_adverserial_v3: (1.0, 0.03612627938241149)
2_adverserial_v4:


#4:

4_naked_v0: (0.6666666666666666, 0.29035640503089677)
4_naked_v1: (0.6178861788617886, 0.26883731865183386)
4_naked_v2: (0.4715447154471545, 0.3765823590863229)
4_naked_v3: (0.35772357723577236, 0.3974809517342973)
4_naked_v4: (0.3170731707317073, 0.39477288450459996)


4_ds_v0: (0.926829268292683, 0.18317829248988451)
4_ds_v1: (0.6585365853658537, 0.27891211480933614)
4_ds_v2: (0.4959349593495935, 0.3541979557380771)
4_ds_v3: (0.8699186991869918, 0.2461335305931693)
4_ds_v4: (0.3008130081300813, 0.44241515468710124)


4_noise_01_v0: (0.7642276422764228, 0.24965981520212896)
4_noise_01_v1: (0.943089430894309, 0.19444602880354953)
4_noise_01_v2: (0.8617886178861789, 0.19448341644680148)
4_noise_01_v3: (0.9105691056910569, 0.20621465980291317)
4_noise_01_v4: (0.9024390243902439, 0.23599099838380683)


4_noise_02_v0: (0.8211382113821138, 0.2585797859500031)
4_noise_02_v1: (0.8780487804878049, 0.20904802689437518)
4_noise_02_v2: (0.943089430894309, 0.20078428847966995)
4_noise_02_v3: (0.9349593495934959, 0.24192132403446912)
4_noise_02_v4: (0.9512195121951219, 0.172837467749505) 


4_noise_03_v0: (0.8699186991869918, 0.20282635190369064)
4_noise_03_v1: (0.9186991869918699, 0.23873642571725517)
4_noise_03_v2: (0.959349593495935, 0.17553320665045163)
4_noise_03_v3: (0.9512195121951219, 0.1764110441960691)
4_noise_03_v4: (0.926829268292683, 0.17881755923439077)


4_noise_04_v0: (0.8943089430894309, 0.19126680009736657)
4_noise_04_v1: (0.983739837398374, 0.15279236800147172)
4_noise_04_v2: (0.9349593495934959, 0.1976159752885933)
4_noise_04_v3: (0.9105691056910569, 0.17611646847260332)
4_noise_04_v4: (0.926829268292683, 0.16672085480531426)


4_repeller_v0: (0.5882352941176471, 0.29488212657790747)
4_repeller_v1: (0.6470588235294118, 0.2557345978413034)
4_repeller_v2: (0.6928104575163399, 0.21124730458830057)
4_repeller_v3: (0.3660130718954248, 0.3615542318719822)
4_repeller_v4: (0.6078431372549019, 0.24401547685085048)


4_types_v0: (0.17886178861788618, 0.4546071904880347)
4_types_v1: (0.24390243902439024, 0.4331471051596904)
4_types_v2: (0.36585365853658536, 0.42840429019105547)
4_types_v3: (0.1951219512195122, 0.4747256046622158)
4_types_v4: (0.34146341463414637, 0.42425736186879576)


4_types_noise_04_v0: (0.9024390243902439, 0.3696188513652769)
4_types_noise_04_v1: (0.8780487804878049, 0.352262248703735)
4_types_noise_04_v2: (0.8617886178861789, 0.3115693293780409)
4_types_noise_04_v3: (0.7154471544715447, 0.35710170485962317)
4_types_noise_04_v4: (0.6097560975609756, 0.37107162652515296)


4_sparse_noise_04_v0: (0.983739837398374, 0.16932136289016878)
4_sparse_noise_04_v1: (0.959349593495935, 0.20415932119235822)
4_sparse_noise_04_v2: (0.9349593495934959, 0.17441658694368803)
4_sparse_noise_04_v3: (0.9512195121951219, 0.1742195820294013)
4_sparse_noise_04_v4: (1.0, 0.17104229645150246)


4_sparse1_noise_04_v0: (0.991869918699187, 0.17959910430318454)
4_sparse1_noise_04_v1: (0.983739837398374, 0.14908433472305846)
4_sparse1_noise_04_v2: (0.991869918699187, 0.1932473448635198)
4_sparse1_noise_04_v3: (0.943089430894309, 0.1887658640165149)
4_sparse1_noise_04_v4: (0.959349593495935, 0.18454545779978818)

4_adverserial_v0: (0.7804878048780488, 0.17430985108719765)
4_adverserial_v1: (0.9349593495934959, 0.17941023093214598)
4_adverserial_v2: (0.7804878048780488, 0.30159168310794654)
4_adverserial_v3: (0.959349593495935, 0.15467334245892267)
4_adverserial_v4: (0.967479674796748, 0.14103840170004572)



20_naked_v0: (0.39335664335664333, 0.19190033711777332)
20_naked_v1: (0.40734265734265734, 0.19686802699245137)
20_naked_v2: (0.34440559440559443, 0.19921504708468532)
20_naked_v3: (0.3409090909090909, 0.19830751400587696)
20_naked_v4: (0.2517482517482518, 0.20210949600838998)


20_ds_v0: (0.21503496503496503, 0.21075054712178956)
20_ds_v1: (0.288, 0.21200178565328368)
20_ds_v2: (0.5, 0.19402739184876183)
20_ds_v3: (0.308, 0.20256596560436818)
20_ds_v4: (0.308, 0.1987081242817012)


20_noise_01_v0: (0.36713286713286714, 0.20149228099303645)
20_noise_01_v1: (0.34, 0.19910752092504524)
20_noise_01_v2: (0.372, 0.1963380911374035)
20_noise_01_v3: (0.42, 0.1857226967616749)
20_noise_01_v4: (0.236, 0.21035112632220035)


20_noise_02_v0: (0.5052447552447552, 0.19400801507407445)
20_noise_02_v1: (0.224, 0.19871079805019518)
20_noise_02_v2: (0.448, 0.1918744037711736)
20_noise_02_v3: (0.268, 0.20039397481095558)
20_noise_02_v4: (0.444, 0.18281824333798652)


20_noise_03_v0: (0.6468531468531469, 0.16894969981297742)
20_noise_03_v1: (0.42, 0.17911248174116998)
20_noise_03_v2: (0.332, 0.19313180924807596)
20_noise_03_v3: (0.328, 0.19260185932648646)
20_noise_03_v4: (0.412, 0.18209938833869632)


20_noise_04_v0: (0.3321678321678322, 0.1901968072103566)
20_noise_04_v1: (0.152, 0.2053412442389766)
20_noise_04_v2: (0.116, 0.2052003312197174)
20_noise_04_v3: (0.236, 0.1961869036087434)
20_noise_04_v4: (0.164, 0.2030803514917294)


20_repeller_v0: (0.21166666666666667, 0.20285098156036369)
20_repeller_v0: (0.344, 0.19663184491854235)
20_repeller_v0: (0.26, 0.201076711861049)
20_repeller_v0: (0.328, 0.19405996022873712)
20_repeller_v0: (0.328, 0.19646663964660338)


20_types_v0: (0.068, 0.21604259766633585)
20_types_v1: (0.076, 0.2183448431531461)
20_types_v2: (0.14, 0.2177983567571351)
20_types_v3: (0.048, 0.21752558243470982)
20_types_v4: (0.088, 0.2167072762343738)


20_types_noise_03_v0: (0.17482517482517482, 0.21452139423881966)
20_types_noise_03_v1: (0.056, 0.22030373280394372)
20_types_noise_03_v2: (0.076, 0.2192249469760734)
20_types_noise_03_v3: (0.06, 0.21873711599295692)
20_types_noise_03_v4: (0.088, 0.22160960611658334)


20_sparse_noise_03_v0: (0.4737762237762238, 0.18141174699734308)
20_sparse_noise_03_v1: (0.3933333333333333, 0.197688751882428)
20_sparse_noise_03_v2: (0.488, 0.17902744548971156)
20_sparse_noise_03_v3: (0.5, 0.1859198257257685)
20_sparse_noise_03_v4: (0.58, 0.18218082985919257)


20_sparse1_noise_03_v0: (0.588, 0.17694291826108127)
20_sparse1_noise_03_v1: (0.584, 0.18038509533006683)
20_sparse1_noise_03_v2: (0.628, 0.17384842811740792)
20_sparse1_noise_03_v3: (0.652, 0.17134160798328515)
20_sparse1_noise_03_v4: (0.636, 0.1733644215401092)
"""

