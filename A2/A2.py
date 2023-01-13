import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

path = "./Dataset/celeba/labels.csv"
data = pd.read_csv(filepath_or_buffer=path, sep='\t')
data = pd.DataFrame(data)


def train_test(train_x, train_y, test_x, test_y):
    train_y = np.squeeze(train_y)
    train_x = train_x / 500
    train_y = train_y
    test_x = test_x / 500

    # Restructuring of 3D data into 2D data
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    # Constructing three SVM models and outputting results
    model = SVC(kernel='linear', C=1)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    print(str(accuracy_score(test_y, y_pred)) + 'for linear kernel SVM')
    model = SVC(kernel='rbf', gamma=0.7, C=1)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    print(str(accuracy_score(test_y, y_pred)) + 'for rbf kernel SVM')
    model = SVC(kernel='poly', degree=3, C=1)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    print(str(accuracy_score(test_y, y_pred)) + 'for poly kernel SVM')
