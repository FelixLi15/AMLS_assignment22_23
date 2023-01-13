import numpy as np
from A1 import A_data as import_data
from A1 import validation_set_extraction as val_data_import_A1
from A1 import A1_1
from A2 import validation_set_extraction_A2 as val_data_import_A2
from A2 import A2
from B1 import B1
from B1 import B1_data
from B1 import B1_data_test
from B2 import KNN
from B2 import B2_1_data
from B2 import B2_1
from B2 import B2_1_data_test


def get_A1_data():
    X, y = import_data.extract_features_labels()
    Y = np.array(y.T)
    tr_X = X[:4000];
    tr_Y = Y[:4000]
    te_X = X[4000:];
    te_Y = Y[4000:]

    return tr_X, tr_Y, te_X, te_Y


def get_A2_data():
    X, y = import_data.extract_features_labels()
    # Y = y.T
    Y = np.array(y.T)
    tr_X = X[:5000];
    tr_Y = Y[:5000]


    return tr_X, tr_Y


def get_B1_data():
    X, y = B1_data.extract_features_labels()
    Y = np.array(y.T)
    tr_X = X[:10000];
    tr_Y = Y[:10000]

    return tr_X, tr_Y


def get_B2_data():
    X, y = B2_1_data.extract_features_labels()
    Y = np.array(y.T)
    tr_X = X[:8000];
    tr_Y = Y[:8000]
    te_X = X[8000:];
    te_Y = Y[8000:]

    return tr_X, tr_Y, te_X, te_Y


def get_A1_validation_data():
    X, y = val_data_import_A1.extract_features_labels()
    # Y = np.array([y, -(y - 1)]).T
    Y = np.array(y.T)
    val_X = X
    val_Y = Y

    return val_X, val_Y


def get_A2_validation_data():
    X, y = val_data_import_A2.extract_features_labels()
    Y = np.array(y.T)
    val_X = X
    val_Y = Y

    return val_X, val_Y


def get_B1_validation_data():
    X, y = B1_data_test.extract_features_labels()
    Y = np.array(y.T)
    val_X = X
    val_Y = Y
    return val_X, val_Y


def get_B2_validation_data():
    X, y = B2_1_data_test.extract_features_labels()
    Y = np.array(y.T)
    val_X = X
    val_Y = Y
    return val_X, val_Y

#Acquisition of data from A1 and training and testing
trainx, trainy, testx, testy = get_A1_data()
val_x, val_y = get_A1_validation_data()
A1_1.train_test(trainx, trainy, testx, testy, val_x, val_y)

#Acquisition of data from A2 and training and testing
trainx, trainy= get_A2_data()
val_x, val_y = get_A2_validation_data()
A2.train_test(trainx, trainy, val_x, val_y)
KNN.KNN_train_test(trainx, trainy, val_x, val_y)

#Acquisition of data from B1 and training and testing
trainx, trainy= get_B1_data()
val_x, val_y = get_B1_validation_data()
B1.train_test(trainx, trainy, val_x, val_y)
KNN.KNN_train_test(trainx, trainy, val_x, val_y)

#Acquisition of data from B2 and training and testing
trainx, trainy, testx, testy = get_B2_data()
val_x, val_y = get_B2_validation_data()
B2_1.train_test(trainx, trainy, testx, testy, val_x, val_y)
