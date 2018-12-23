import pickle, pprint
#hardcode datapath to this char_trajectories
#DATA_FILE =


def load_data(file='../RNN/data/char_trajectories.pkl'):
    pkl_file = open(file, 'rb')

    data = pickle.load(pkl_file)
    #pprint.pprint(data)

    pkl_file.close()
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]

    return((X_train, Y_train), (X_test, Y_test))


