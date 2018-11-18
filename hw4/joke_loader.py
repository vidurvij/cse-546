import numpy as np

def data_load():
    m = 24983
    n = 100
    train_m = np.zeros((m,n))
    print(train_m.shape)
    train  = open("/home/vidur/Desktop/cse 546/hw4/jokes/train.txt",'r')
    for i in train:
        i = i.strip()
        # print(i)
        idx,idy,d = i.split(',')
        # print(train_m.shape)
        train_m[int(idx)-1,int(idy)-1] = d
    train.close()
    test = open("/home/vidur/Desktop/cse 546/hw4/jokes/test.txt",'r')
    test_m = np.zeros((m,n))
    for i in test:
        i = i.strip()
        idx, idy, d =  i.split(',')
        test_m[int(idx)-1][int(idy)-1] = d
    test.close
    # print(train_m)
    return train_m, test_m
