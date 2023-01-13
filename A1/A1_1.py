from sklearn.neural_network import MLPClassifier


def train_test(X_train, trainy, X_test, testy,val_x, val_y):
    #Restructuring of 3D data into 2D data to meet the input requirements of the MLPClassifier
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) / 500
    X_test = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])) / 500
    val_x = (val_x.reshape(val_x.shape[0], val_x.shape[1] * val_x.shape[2])) / 500

    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(10,), batch_size=200, max_iter=1600)
    clf.fit(X_train, trainy)
    score = clf.score(X_test, testy, sample_weight=None)
    print(' Test Accuracy:', score)
    print(' Val Accuracy:', clf.score(val_x, val_y, sample_weight=None))
