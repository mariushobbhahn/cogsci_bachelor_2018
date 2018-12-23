import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


#pad the sequences with zeros such that all have the same length
def pad_sequence(seq, length):
    if len(seq) < length:
        seq = np.append(seq, np.zeros(length - len(seq)))
        #print(type(seq))
    return seq

#if take the cumulative sum of the deltas to save absolute values
def cumulative_sum_seq(sequence):
    sequence[0], sequence[1] = np.cumsum(sequence[0]), np.cumsum(sequence[1])
    return(sequence)

#method to visualize any image:
def plot_sequence(index, sequences, swapaxis=False, deltas=False):
    if swapaxis:
        sequences = np.swapaxes(sequences,1 ,2)
    if deltas:
        x,y = np.cumsum(sequences[index][0], sequences[index][1])
    else:
        x,y = sequences[index][0], sequences[index][1]
    plt.plot(x, y)
    plt.show()

#lastly we want to normalize the sequences. We choose to normalize such that values of a sequence lie between -0.5 and 0.5
def standardize_data(data, separate=False):
    """Brings the mean of the data set to 0 and the standard deviation to 1.

    # Arguments:
        data: Set of trajectories as ndarray of shape [numberOfTrajectories, trajectoryLenth, x/y].
        separate: Whether to apply normalization for each coordinate dimension separately.
    # Returns:
        Standardized ndarray of trajectories.
    """
    if(separate):
        x_mean = np.mean(data[:,:,0])
        y_mean = np.mean(data[:,:,1])
        x_stdev = np.std(data[:,:,0])
        y_stdev = np.std(data[:,:,1])
        # Standardize all sequences
        data[:,:,0] = (data[:,:,0] - x_mean) / x_stdev
        data[:,:,1] = (data[:,:,1] - y_mean) / y_stdev
        return data
    else:
        mean = np.mean(data)
        stdev = np.std(data)
        data = (data - mean) / stdev
        return data


#to make it a sequence and not a oneshot guess
def pad_class(class_vector, x=1, num_classes=20):
    x_r = 205 - x #number of timesteps to fill
    class_vector = np.tile(class_vector, x)
    fill_vector = np.tile(np.zeros(num_classes),(x_r,1))
    full_vector = np.vstack((class_vector, fill_vector))
    return(full_vector)

#for class preparation
def char_to_one_hot(char, num_of_chars, list_of_chars):
    one_hot_vec = np.zeros(num_of_chars)
    index_of_char = list_of_chars.index(char)
    if index_of_char < num_of_chars:
        one_hot_vec[index_of_char] = 1
        return(one_hot_vec)
    else:
        return(None)

#not actually used
def shuffle_data(data, targets):
    assert len(data) == len(targets)
    p = np.random.permutation(len(data))
    return(data[p], targets[p])

#final function
def process_character_trajectories(pad_sequences=True,
                                   cumulative_sum=True,
                                   gen_sequences=True,
                                   standardize=True,
                                   shuffle=True,
                                   test_size=0.2,
                                   num_classes=20,
                                   noise=False,
                                   noise_std=0.1,
                                   clip=True,
                                   repeller=False,
                                   filename='char_trajectories.p kl'):
    #load data
    mat_contents = sio.loadmat('mixoutALL_shifted.mat')
    consts = mat_contents['consts']
    sequences = mat_contents['mixout'][0]


    #we want all the sequences to have the same length,
    #so we add zeros for padding in the end.
    MAX_LEN = max([len(seq[0]) for seq in sequences])

    if pad_sequences:
        sequences_final = np.zeros((2858, 2, MAX_LEN))

        #pad dimension x and y
        for i in range(len(sequences)):
            sequences_final[i][0] = pad_sequence(sequences[i][0], MAX_LEN)
            sequences_final[i][1] = pad_sequence(sequences[i][1], MAX_LEN)

        #print result to show it has been padded
        print("shape of sequences final", np.shape(sequences_final))


    if cumulative_sum:
        sequences_final = [cumulative_sum_seq(seq) for seq in sequences_final]

    if gen_sequences:
    #swap axis such that shape is (205,2) not (2,205)
        print("shape before swap: ",np.shape(sequences_final))
        sequences_final = np.swapaxes(sequences_final, axis1=1, axis2=2)
        print("shape after swap: ",np.shape(sequences_final))

    #plot random letter before standardizing
    #plot_sequence(300, sequences_final, swapaxis=True)
    if standardize:
        sequences_final = standardize_data(sequences_final, separate=True)
        #plot same letter after standardizing
        #plot_sequence(300, sequences_final, swapaxis=True)

    #now we prepare classes:
    classes = []
    #last letter of kind
    # index of last letter of kind
    classes.extend(['a'] * (97 - 0))  # a = 96
    classes.extend(['b'] * (170 - 97))  # b = 169
    classes.extend(['c'] * (225 - 170))  # c = 224
    classes.extend(['d'] * (307 - 225))  # d = 306
    classes.extend(['e'] * (420 - 307))  # e = 419
    classes.extend(['g'] * (486 - 420))  # g = 485
    classes.extend(['h'] * (543 - 486))  # h = 542
    classes.extend(['l'] * (623 - 543))  # l = 622
    classes.extend(['m'] * (692 - 623))  # m = 691
    classes.extend(['n'] * (748 - 692))  # n = 747
    classes.extend(['o'] * (816 - 748))  # o = 815
    classes.extend(['p'] * (886 - 816))  # p = 885
    classes.extend(['q'] * (956 - 886))  # q = 955
    classes.extend(['r'] * (1013 - 956))  # r = 1012
    classes.extend(['s'] * (1077 - 1013))  # s = 1076
    classes.extend(['u'] * (1144 - 1077))  # u = 1143
    classes.extend(['v'] * (1218 - 1144))  # v = 1217
    classes.extend(['w'] * (1278 - 1218))  # w = 1277
    classes.extend(['y'] * (1345 - 1278))  # y = 1344
    classes.extend(['z'] * (1433 - 1345))  # z = 1432
    classes.extend(['a'] * (1507 - 1433))  # a = 1506
    classes.extend(['b'] * (1575 - 1507))  # b = 1574
    classes.extend(['c'] * (1662 - 1575))  # c = 1661
    classes.extend(['d'] * (1737 - 1662))  # d = 1736
    classes.extend(['e'] * (1810 - 1737))  # e = 1809
    classes.extend(['g'] * (1882 - 1810))  # g = 1881
    classes.extend(['h'] * (1952 - 1882))  # h = 1951
    classes.extend(['l'] * (2046 - 1952))  # l = 2045
    classes.extend(['m'] * (2102 - 2046))  # m = 2101
    classes.extend(['n'] * (2176 - 2102))  # n = 2175
    classes.extend(['o'] * (2249 - 2176))  # o = 2248
    classes.extend(['p'] * (2310 - 2249))  # p = 2309
    classes.extend(['q'] * (2364 - 2310))  # q = 2363
    classes.extend(['r'] * (2425 - 2364))  # r = 2425
    classes.extend(['s'] * (2495 - 2426))  # s = 2494
    classes.extend(['u'] * (2559 - 2495))  # u = 2558
    classes.extend(['v'] * (2640 - 2559))  # v = 2639
    classes.extend(['w'] * (2705 - 2640))  # w = 2704
    classes.extend(['y'] * (2775 - 2705))  # y = 2774
    classes.extend(['z'] * (2858 - 2775))  # z = 2857


    #these are all possible letters:
    list_of_chars = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']

    #select only the first num_classes classes
    classes_new = []
    sequences_new = []
    for i in range(len(classes)):
        char = classes[i]
        index_of_char = list_of_chars.index(char)
        if index_of_char < num_classes:
            one_hot_vector = np.zeros(num_classes)
            one_hot_vector[index_of_char] = 1
            classes_new.append(one_hot_vector)
            sequences_new.append(sequences_final[i])

    classes_final = np.array(classes_new)
    sequences_final = np.array(sequences_new)
    print("classes_final: ", np.shape(classes_final))
    print("sequences_final: ", np.shape(sequences_final))

    #we want to add a uniform class that projects onto a vector where all coordinates are (0,0).
    #this should repell the network from classifying something as uniform or close to uniform and hopefully forces it to decide more clearly for a class
    if repeller:
        #produce a uniformly distributed class vector:
        uniform_class_vector = np.array([1 / num_classes] * num_classes)
        #we construct an empty target:
        empty_target = np.zeros(shape=(205,2))
        #we pretend as if the uniform class vector was a class by itself and add that many vectors:
        uniform_class_vectors = np.tile(uniform_class_vector, (int(sequences_final.shape[0]/num_classes), 1))
        print("shape of uniform class vectors: ", np.shape(uniform_class_vectors))
        empty_targets = np.tile(empty_target, (int(sequences_final.shape[0]/num_classes), 1, 1))
        print("shape of empty targets: ", np.shape(empty_targets))
        #and add them to the classes
        classes_final = np.concatenate((classes_final, uniform_class_vectors))
        sequences_final = np.concatenate((sequences_final, empty_targets))
        print("shape of classes after adding repeller: ", np.shape(classes_final))
        print("shape of sequences after adding repeller: ", np.shape(sequences_final))

    #to make the model more robust we add noise to have a bigger range of values that can be a certain letter
    if noise:
        noise_array = np.random.normal(loc=0, scale=noise_std, size=(classes_final.shape[0], classes_final.shape[1]))
        assert(noise_array.shape == classes_final.shape)
        classes_final += noise_array

    if clip:
        classes_final = np.clip(classes_final, a_min=0, a_max=1)


    #if we want to generate sequences we also need to adapt our classes
    # since we want this input only at a number x of early timesteps, we provide num_classes-vectors of zeros for the latter 205-x entries
    if gen_sequences:
        print("shape of class padding: ", np.shape(pad_class(classes_final[0], num_classes=num_classes)))
        classes_final = np.array([pad_class(class_vector, num_classes=num_classes) for class_vector in classes_final])
        #print shape of classes after padding
        #print("shape of classes:", np.shape(classes))


    #since our data is ordered we need to split it in train and test split. Validation will be done by keras
    train_classes, test_classes, train_sequences, test_sequences = train_test_split(classes_final, sequences_final, test_size=test_size, shuffle=shuffle)

    #print shapes of data:
    print("shape of X_train: ", np.shape(train_classes))
    print("shape of Y_train: ", np.shape(train_sequences))
    print("shape of X_test: ", np.shape(test_classes))
    print("shape of Y_test: ", np.shape(test_sequences))

    """
    print("X_train[10]: ", train_classes[10][0])
    print("X_train[11]: ", train_classes[11][0])
    print("X_train[12]: ", train_classes[12][0])
    print("X_train[13]: ", train_classes[13][0])
    print("X_train[14]: ", train_classes[14][0])
    print("X_train[15]: ", train_classes[15][0])
    plot_sequence(10, train_sequences, swapaxis=True, deltas=False)
    plot_sequence(11, train_sequences, swapaxis=True, deltas=False)
    plot_sequence(12, train_sequences, swapaxis=True, deltas=False)
    plot_sequence(13, train_sequences, swapaxis=True, deltas=False)
    plot_sequence(14, train_sequences, swapaxis=True, deltas=False)
    plot_sequence(15, train_sequences, swapaxis=True, deltas=False)
    """
    #save data in a file
    all_data = np.array([train_classes, train_sequences, test_classes, test_sequences])
    output = open(filename, 'wb')
    pickle.dump(all_data, output)
    output.close()



process_character_trajectories(num_classes=2,
                               noise=False,
                               filename='char_trajectories_2_repeller.pkl',
                               repeller=True)

