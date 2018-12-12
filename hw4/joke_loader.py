import numpy as np

def data_load(matrix = False):
    n = 1000
    m = 500
    # if matrix:
    train_mr = np.zeros((n,m))
    # print(train_m.shape)
    train  = open("/home/vidur/Desktop/cse 546/hw4/jokes/train.txt",'r')
    for i in train:
        i = i.strip()
        # print(i)
        a = i.split(',')
        a[0]  = int(a[0])
        a[1]  = int(a[1])
        a[2]  = float(a[2])
        # print(type(a[0]))
        # print(train_m.shape)
        train_mr[a[0]-1,a[1]-1] = a[2]
    train.close()
        # print(train_m)
    # if not matrix:
    train_m = []
    # print(train_m.shape)
    train  = open("/home/vidur/Desktop/cse 546/hw4/jokes/train.txt",'r')
    for i in train:
        i = i.strip()
        # print(i)
        a = i.split(',')
        a[0]  = int(a[0])
        a[1]  = int(a[1])
        a[2]  = float(a[2])
        # a[1]  = int(a[1])
        # print(train_m.shape)
        train_m.append(a)
    train.close()
    test = open("/home/vidur/Desktop/cse 546/hw4/jokes/test.txt",'r')
    test_m = []
    for i in test:
        i = i.strip()
        a =  i.split(',')
        a[0]  = int(a[0])
        a[1]  = int(a[1])
        a[2]  = float(a[2])
        # a = np.array([idx,idy,d])
        test_m.append(a)
    test.close
        # print(train_m)
    if not matrix:
        return np.array(train_m), np.array(test_m), n ,m
    if matrix:
        return train_mr,np.array(train_m), np.array(test_m), n ,m
# train, test, n, m = data_load(True)
# print(train.shape)
# for i in range(train.shape[1]):
#     s = np.count_nonzero(train[:,i])
#     assert s !=0
