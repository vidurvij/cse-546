import numpy as np
from mnist import MNIST
import multiprocessing as mp
from itertools import cycle
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_data():
    mndata = MNIST()
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, X_test, labels_test, labels_train

def distance(centers, point):
    # c = np.copy(centers)
    # print("@@",type(centers))
    # print("@@",type(c))
    # print(point)Fdis
    # point = np.array
    # print((centers),type(point))
    # print(type(c))
    # print(c.shape,point.shape)
    distances = np.sum(np.square(np.subtract(centers , point)), axis = 1)
    cluster = np.argmin(distances)
    return np.min(distances), cluster

def Super(train, d):
    center = np.zeros((d,train.shape[1]))
    distances = np.zeros((train.shape[0]))
    # print()
    center[0] = train[np.random.choice(np.arange(train.shape[0]))]
    # print((train[np.random.choice(np.arange(train.shape[0]))]).shape)
    probability = np.zeros((train.shape[0],1))
    # print(center[range(6),:].shape)0
    # exit()
    for i in range(1,d):
        l = center[range(i),:]
        for j in range(train.shape[0]-1,-1,-1):
            # print(j)
            # print(type(l))
            # print(train[j].shape)
            distances[j], c = distance(l,train[j])
        probability = distances**2/np.sum(distances**2)
        center[i] = train[np.random.choice(np.arange(train.shape[0]), p = probability )]
    return center

if __name__ == "__main__" :
    super = False
    train, test, lt, lt2 = load_data()

    clusters = np.zeros(train.shape[0], dtype = int)
    # centers = np.random.randn(train.shape[0], train.shape[1])
    i = 7
    # train = train.tolist()
    # while i < train.shape[0] :
    for d in tqdm([5, 10, 15]) :
        centers = np.random.random((d, train.shape[1]))
        lold = 0
        loss = []
        total = 100000
        if super:
            centers = Super(train,d)
        # for i in range(500) :
        print(lold-total)
        while abs(lold-total) > 0:
            lold = total
            total = 0
            for i in range(train.shape[0]):
                dr, clusters[i] = distance(centers,train[i])
                total += dr
            print(total)
            loss.append(total)
            # print(d)
            cum = np.zeros((d, train.shape[1]))
            count = np.zeros((d,1))
            # print(clusters[i])
            for i in range(train.shape[0]):
                cum[clusters[i]] += train[i]
                count[clusters[i]] += 1
            centers = np.divide(cum,count)
        plt.plot(loss,".-", label = str(d)+" Clusters")
    plt.grid()
    plt.label()
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Distance")
    plt.show()
    if super:
        plt.savefig("K-Means ++")
    if not super:
        plt.savefig("K-Means")
