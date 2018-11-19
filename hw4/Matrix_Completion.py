import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
class Matrix_Completion():
    def __init__(self, train, test,n,m, algorithm = "SGD"):
        permutations = np.random.permutation(train)
        self.train = permutations[0:int(.8*train.shape[0])]
        self.valid = permutations[int(.8*train.shape[0]):train.shape[0]]
        self.test = test
        self.u = 0
        self.v = 0
        self.m = m
        self.n = n
        self.lam = .3
        self. alpha = .01
        self.algorithm = algorithm

    def Regression(self, d = None, lam = 0):
        if d != 1:
            lam = self.lam
        self.u = np.random.randn(self.n,d)
        self.v = np.random.randn(self.m,d)
        for j in range(1000):
            # while error > .01:
            # print(( 2 * self.lam * np.sum(2*self.u,0).T).shape)
            i = np.random.permutation(self.train)[j]
            # print(type(i[0]))
            delv = 2 * self.u[int(i[0])-1].T * (self.u[int(i[0])-1]@self.v[int(i[1])-1].T-i[2]) + 2 * lam * self.v[int(i[1])-1]
            delu = 2 * self.v[int(i[1])-1] * (self.u[int(i[0])-1]@self.v[int(i[1])-1].T-i[2]) + 2 * lam * self.u[int(i[0])-1].T
            # print(delu.shape)
            # print(np.linalg.norm(self.u),np.linalg.norm(self.v))
            self.u[int(i[0])-1] =- self.alpha * delu
            self.v[int(i[1])-1] =- self.alpha * delv
            # print(v)
            error = self.error_scoring(1,self.u,self.v)
            print(error)
            # errors.append(error)

    def error_scoring(self,set, u = 0, v = 0):
        # print (u)
        # exit()
        if isinstance(u,int):
            u = self.u
            v = self.v
        square = 0
        absolute = np.zeros((self.n,1))
        if set == 0:
            for idx in range(test.shape):
                i = self.test[idx]
                # print(i)
                s_hat = u[int(i[0])-1]@v[int(i[1])-1].T
                s = i[2]
                # print("@@@@@@@@@@@@@@@@@@@@@@@",type(s))
                # exit()
                square += (s-s_hat)**2
                absolute[int(i[0])-1] += np.absolute(s_hat-s)
            square = square/self.test.shape[0]
        if set == 1:
            for idx in range(self.valid.shape[0]):
                # print(u.shape,v.shape)
                i = self.valid[idx]
                # print(i)
                s_hat = u[int(i[0])-1]@v[int(i[1])-1].T
                s = i[2]
                # print("@@@@@@@@@@@@@@@@@@@@@@@",type(s))
                # exit()
                square += (s-s_hat)**2
                absolute[int(i[0])-1] += np.absolute(s_hat-s)
            square = square/self.valid.shape[0]
        absolute = np.mean(square)
        return square, absolute

    def cross_validation(self):
        errors = []
        d_range = [50]
        for d in d_range:
            self.u = np.random.randn(self.n,d)
            self.v = np.random.randn(self.m,d)
            # print((self.v[int(8)-1]).shape)
            # while error > .01:
            for j in range(100):
                # print(( 2 * self.lam * np.sum(2*self.u,0).T).shape)
                i = np.random.permutation(self.train)[j]
                # print(type(i[0]))
                delv = 2 * self.u[int(i[0])-1].T * (self.u[int(i[0])-1]@self.v[int(i[1])-1].T-i[2]) + 2 * self.lam * self.v[int(i[1])-1]
                delu = 2 * self.v[int(i[1])-1] * (self.u[int(i[0])-1]@self.v[int(i[1])-1].T-i[2]) + 2 * self.lam * self.u[int(i[0])-1].T
                # print(delu.shape)
                # print(np.linalg.norm(self.u),np.linalg.norm(self.v))
                self.u[int(i[0])-1] =- self.alpha * delu
                self.v[int(i[1])-1] =- self.alpha * delv
                # print(v)
                error = self.error_scoring(1,self.u,self.v)
                print(error)
                errors.append(error)
        plt.plot(errors)
        plt.savefig("Error.png")


def SVD(train,test):
    d_range = [1,2,5,10,20,50]
    errors = []
    errorst = []
    for d in d_range:
        a, b,c = svds(train,d)
        # print(a.shape, b.shape, c.shape)
        tr = a@np.diag(b)@c
        error_train = np.sum(np.square(np.abs(tr-train)))/np.count_nonzero(train)
        error_test = 0

        for i in test:
            # print(type(i[2]),type(tr[1,2]))
            error_test += np.square(np.abs(i[2] - tr[int(i[0])-1, int(i[1]) - 1]))
        error_test /= test.shape[0]
        # print(error_train.shape)
        # exit()
        errors.append(error_train)
        errorst.append(error_test)
    plt.plot(errors,"-.")
    plt.plot(errorst,"-.")
    plt.grid()
    plt.yticks(d_range)
    plt.show()
