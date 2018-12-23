import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


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
def plot_sequence(index, sequences, filename='char', title='', save=False, plot=True, swapaxis=False, deltas=False):
    if swapaxis:
        sequences = np.swapaxes(sequences,1 ,2)
    if deltas:
        x,y = np.cumsum(sequences[index][0], sequences[index][1])
    else:
        x,y = sequences[index][0], sequences[index][1]
    plt.title(title)
    plt.plot(x, y)
    if plot:
        plt.show()
    if save:
        plt.savefig('../data/characters/{}'.format(filename))
        plt.clf()


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

#function that transforms a given number of letters to a cluster
def cluster_chars(sequences, num_cluster, show_centers=False, labels_to_one_hot=True, print_labels=False):
    sequences = np.reshape(sequences, newshape=(len(sequences), 410))
    kmeans = KMeans(n_clusters=num_cluster, )
    kmeans = kmeans.fit(sequences)
    labels = kmeans.predict(sequences)
    c = kmeans.cluster_centers_

    if show_centers:
        # show the different classes
        c = np.reshape(c, newshape=(len(c), 205, 2))
        plot_sequence(index=0, sequences=c, swapaxis=True)
        plot_sequence(index=1, sequences=c, swapaxis=True)
        plot_sequence(index=2, sequences=c, swapaxis=True)
        plot_sequence(index=3, sequences=c, swapaxis=True)

    #reshape the sequences such that they are 2D again
    sequences = np.reshape(sequences, newshape=(len(sequences), 205, 2))
    # reshape the labels to one-hot vectors:
    if labels_to_one_hot:
        labels = np.array([cluster_to_one_hot(label, num_cluster) for label in labels])
    if print_labels:
        print("labels: ", labels)

    return(labels, sequences)


#every cluster can be represented as a one-hot vector:
def cluster_to_one_hot(cluster, num_cluster):
    one_hot = np.zeros(num_cluster)
    one_hot[cluster] = 1
    return(one_hot)

#function that pads an array of clustered character classes
def pad_clustered_char(clustered_char_array, sequences, index_of_char, num_classes, num_cluster):
    multi_dim_one_hot = np.zeros(shape=(num_classes, num_cluster))
    padded_classes = []
    for i in range(len(clustered_char_array)):
        multi_dim_one_hot[index_of_char] = clustered_char_array[i]
        padded_classes.append(multi_dim_one_hot)

    return(np.array(padded_classes), sequences)

#main function:
def load_character_trajectories_cluster(pad_sequences=True,
                                        cumulative_sum=True,
                                        gen_sequences=True,
                                        standardize=True,
                                        test_size=0.2,
                                        shuffle=True,
                                        num_classes=20,
                                        clustering=True,
                                        num_cluster=4,
                                        ):

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
    #index of last letter of kind
    classes.extend(['a'] * (97 - 0))#a = 96
    classes.extend(['b'] * (170 - 97))#b = 169
    classes.extend(['c'] * (225 - 170))#c = 224
    classes.extend(['d'] * (307 - 225))#d = 306
    classes.extend(['e'] * (420 - 307))#e = 419
    classes.extend(['g'] * (486 - 420))#g = 485
    classes.extend(['h'] * (543 - 486))#h = 542
    classes.extend(['l'] * (623 - 543))#l = 622
    classes.extend(['m'] * (692 - 623))#m = 691
    classes.extend(['n'] * (748 - 692))#n = 747
    classes.extend(['o'] * (816 - 748))#o = 815
    classes.extend(['p'] * (886 - 816))#p = 885
    classes.extend(['q'] * (956 - 886))#q = 955
    classes.extend(['r'] * (1013 - 956))#r = 1012
    classes.extend(['s'] * (1077 - 1013))#s = 1076
    classes.extend(['u'] * (1144 - 1077))#u = 1143
    classes.extend(['v'] * (1218 - 1144))#v = 1217
    classes.extend(['w'] * (1278 - 1218))#w = 1277
    classes.extend(['y'] * (1345 - 1278))#y = 1344
    classes.extend(['z'] * (1433 - 1345))#z = 1432
    classes.extend(['a'] * (1507 - 1433))#a = 1506
    classes.extend(['b'] * (1575 - 1507))#b = 1574
    classes.extend(['c'] * (1662 - 1575))#c = 1661
    classes.extend(['d'] * (1737 - 1662))#d = 1736
    classes.extend(['e'] * (1810 - 1737))#e = 1809
    classes.extend(['g'] * (1882 - 1810))#g = 1881
    classes.extend(['h'] * (1952 - 1882))#h = 1951
    classes.extend(['l'] * (2046 - 1952))#l = 2045
    classes.extend(['m'] * (2102 - 2046))#m = 2101
    classes.extend(['n'] * (2176 - 2102))#n = 2175
    classes.extend(['o'] * (2249 - 2176))#o = 2248
    classes.extend(['p'] * (2310 - 2249))#p = 2309
    classes.extend(['q'] * (2364 - 2310))#q = 2363
    classes.extend(['r'] * (2425 - 2364))#r = 2425
    classes.extend(['s'] * (2495 - 2426))#s = 2494
    classes.extend(['u'] * (2559 - 2495))#u = 2558
    classes.extend(['v'] * (2640 - 2559))#v = 2639
    classes.extend(['w'] * (2705 - 2640))#w = 2704
    classes.extend(['y'] * (2775 - 2705))#y = 2774
    classes.extend(['z'] * (2858 - 2775))#z = 2857


    #these are all possible letters:
    list_of_chars = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']

    """
    for i in range(len(classes)):
        plot_sequence(index=i, title='index_{} '.format(i), filename='char_{}_index_{}'.format(classes[i], i), save=True, plot=False, sequences=sequences_final, swapaxis=True)
        print("class_{}: ".format(i), classes[i])
    """

    #for i in range(1425, 1510):
    #    plot_sequence(index=i, title='index_{} '.format(i), sequences=sequences_final, swapaxis=True)

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

    #show that only the select classes are represented
    classes_final = np.array(classes_new)
    sequences_final = np.array(sequences_new)
    print("classes_final: ", np.shape(classes_final))
    print("sequences_final: ", np.shape(sequences_final))

    #we cluster each character seperately
    #first we get all characters that belong to the same class
    characters = [0] * num_classes #list of zeros with length of num_classes
    for i in range(num_classes):
        all_letter_classes = []
        all_letter_sequences = []
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[i] = 1
        for j in range(len(classes_final)):
            if np.array_equal(classes_final[j], one_hot_vector):
                all_letter_classes.append(classes_final[j])
                all_letter_sequences.append((sequences_final[j]))

        #print("all_letter_classes: ", all_letter_classes)
        characters[i] = tuple([all_letter_classes, all_letter_sequences])

    characters = np.array(characters)
    print("characters: ", np.shape(characters))

    #secondly, we cluster each letter individually
    characters_clustered = [cluster_chars(sequences=tup[1], num_cluster=num_cluster, show_centers=False) for tup in characters]
    print("characters clustered: ", np.shape(characters_clustered))
    classes_clustered_padded = []
    for i in range(len(characters_clustered)):
        tup = characters_clustered[i]
        classes_clustered_padded.append(pad_clustered_char(tup[0], sequences=tup[1], index_of_char=i, num_classes=num_classes, num_cluster=num_cluster))

    classes_clustered_padded = np.array(classes_clustered_padded)
    #print("characters_clustered_padded: ", classes_clustered_padded)

    #thirdly we try put all the tuples back together as classes
    classes_final = classes_clustered_padded[:,0]
    sequences_final =  classes_clustered_padded[:,1]

    print("classes_final: ", np.shape(classes_final))
    print("sequences_final: ", np.shape(sequences_final))


    # since our data is ordered we need to split it in train and test split. Validation will be done by keras
    train_classes, test_classes, train_sequences, test_sequences = train_test_split(classes_final, sequences_final,
                                                                                    test_size=test_size,
                                                                                    shuffle=shuffle)

    # print shapes of data:
    print("shape of X_train: ", np.shape(train_classes))
    print("shape of Y_train: ", np.shape(train_sequences))
    print("shape of X_test: ", np.shape(test_classes))
    print("shape of Y_test: ", np.shape(test_sequences))


load_character_trajectories_cluster(num_classes=4)