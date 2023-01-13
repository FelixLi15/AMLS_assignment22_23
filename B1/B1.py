from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def train_test(X_train, y_train, X_test, y_test):
    # Restructuring of 3D data into 2D data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    # Construction of a random forest with 60 decision trees
    n = 60
    clf = RandomForestClassifier(n_estimators=n)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # prediction on test set
    y_pred = clf.predict(X_test)

    print("Random Forest test Accuracy:", accuracy_score(y_test, y_pred))
