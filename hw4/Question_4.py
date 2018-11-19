import numpy as np
from mnist import MNIST
import multiprocessing as mp
from itertools import cycle

def load_data():
    mndata = MNIST()
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, X_test, labels_test, labels_train

def distance(centers, point):
    # print(point)
    # point = np.array
    distance = np.sum(np.square(centers - point), axis = 1)
    cluster = np.argmin(distance)
    return np.min(distance), cluster

if __name__ == "__main__" :
    train, test, lt, lt2 = load_data()

    clusters = np.zeros(train.shape[0], dtype = int)
    # centers = np.random.randn(train.shape[0], train.shape[1])
    i = 7
    # train = train.tolist()
    # while i < train.shape[0] :
    for d in [5] :

        centers = np.random.random((d, train.shape[1]))
        for i in range(500) :
            total = 0
            for i in range(train.shape[0]):
                dr, clusters[i] = distance(centers,train[i])
                total += dr
            print(total)
            # print(d)
            cum = np.zeros((d, train.shape[1]))
            count = np.zeros((d,1))
            # print(clusters[i])
            for i in range(train.shape[0]):
                cum[clusters[i]] += train[i]
                count[clusters[i]] += 1
            centers = np.divide(cum,count)
