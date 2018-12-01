import torch as tt
import numpy as np
import matplotlib.pyplot as plt


def onehot(x,label):
    xhot = tt.zeros(x.shape[0],len(label))
    for i in range(xhot.shape[0]):
        xhot[i][x[i]] = 1
    return xhot

def Data_Generation():
    n = 50
    xs = []
    ys = []
    for i in range(1,n+1,1):
        f = 0
        # print(i)
        x = i/(n-1)
        xs.append(x)
        for k in range(1,5,1):
            # print(k)
            if x >= (k/5):
                f += 1
        # print(f)
        f *=10
        if i == 25:
            y = 0
        else:
            y = f + np.random.normal()
        ys.append(y)
    return np.array(xs).reshape(n,1), np.array(ys).reshape(n,1)

def D_matrix(n):
    D = np.zeros((n-1,n))
    for i in range(n-1):
        for j in range(n):
            if i == j:
                D[i][j] = -1
            if i == j-1:
                D[i][j] = 1
    return D

class Plotter():
    def __init__(self,x,y):
        self.x = x
        self.len = x.shape[0]
        self.y = y
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.plot(self.x,self.y,label = "Orginal Data")
    def overlay(self,alpha,xtrain):
        yp = np.zeros((self.len,1))
        for i in range(self.len):
            yp[i] = np.sum(alpha*Ker(self.x[i],xtrain))
        self.ax.plot(self.x,yp,label = "Predicted")
        # self.fig.show()

def Ker(x,z,gam = 500):
    # print(x,z)
    res = np.exp(-gam *((x-z)**2))
    return res

def Km(x):
    K = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            K[i][j] = Ker(x[i],x[j])
    return K
