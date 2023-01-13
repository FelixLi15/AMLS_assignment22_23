import pandas as pd
import numpy as np

basedir = 'D:/AMLS_22-23 _SN22074364/Dataset/cartoon_set1/img/'


def knn_classify(new_data, X, label, k):
    """
    :param new_data: Data to be forecast
    :param X: Raw feature data
    :param label: Label data
    :param k: k value
    :return: Result
    """
    # Build the original dataset that has been sorted -dataSet
    result = []
    # Calculate the distance between a point in a known category dataset and the current point
    dist = (np.sum((X - new_data) ** 2, axis=1)) ** 0.5
    # Arrange the distances in ascending order, then select the k points with the smallest distance
    dist_up = pd.DataFrame({"Distance": dist, "Feature": label})
    d_k_neighbor = dist_up.sort_values(by="Distance")[: k]
    # Determine the frequency of occurrence of the category in which the first k points are located
    res = d_k_neighbor.loc[:, "Feature"].value_counts()
    # Select the category with the highest frequency as the predicted category for the current point
    result.append(res.index[0])
    return result

def KNN_train_test(train_x, train_y, val_x, val_y):
    # Collation of data to meet KNN input requirements
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    val_x = val_x.reshape(val_x.shape[0], val_x.shape[1] * val_x.shape[2])
    train_x = train_x / 500
    val_x = val_x / 500
    train_y = train_y
    val_y = val_y

    #Initialisation of experimental results,k=5
    m = 0
    n = 0
    right=0
    for data in val_x:
        result = knn_classify(data, train_x, train_y, 5)
        if result == val_y[m]:
            n = n + 1
        m = m+1
        right = n / val_y.shape[0]
    print('When k = 5, KNN test Accuracy: ')
    print(right)
    print('\n')

    # Initialisation of experimental results,k=11
    m = 0
    n = 0
    right = 0
    for data in val_x:
        result = knn_classify(data, train_x, train_y, 11)
        if result == val_y[m]:
            n = n + 1
        m = m+1
        right = n / val_y.shape[0]
    print('When k = 11, KNN test Accuracy: ')
    print(right)
