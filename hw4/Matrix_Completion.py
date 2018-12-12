import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from tqdm import tqdm
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
        self.lam = [64, 120, 130, 140, 240, 256]
        self. alpha = .01
        self.algorithm = algorithm

    def Regression(self, d = None, lam = 0):
        if d != 1:
            lam = self.lam
        self.u = np.random.randn(self.n,d)
        self.v = np.random.randn(self.m,d)
        for j in range(100):
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
        print(self.u)

    def error_scoring(self,set, u = 0, v = 0):
        # print (u)
        # exit()
        if isinstance(u,int):
            u = self.u
            v = self.v
        square = 0
        absolute = np.zeros((self.n,1))
        if set == 0:
            for idx in range(self.test.shape[0]):
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
        if set == 2:
            for idx in range(self.train.shape[0]):
                # print(u.shape,v.shape)
                i = self.train[idx]
                # print(i)
                s_hat = u[int(i[0])-1]@v[int(i[1])-1].T
                s = i[2]
                # print("@@@@@@@@@@@@@@@@@@@@@@@",type(s))
                # exit()
                square += (s-s_hat)**2
                absolute[int(i[0])-1] += np.absolute(s_hat-s)
            square = square/self.train.shape[0]
        absolute = np.mean(square)
        return square

    # def cross_validation_sgd(self):
    #     errors = []
    #     d_range = [50]
    #     old_error = np.array(100000)
    #     error = 0
    #     for d in d_range:
    #         print(d)
    #         normu = 100000
    #         normv = 100000
    #         self.u = np.random.randn(self.n,d)
    #         self.v = np.random.randn(self.m,d)
    #         # print((self.v[int(8)-1]).shape)
    #         # while error > .01:
    #         while normu > 10**-15 or normv> 10**-15  :
    #             oldu = np.copy(self.u)
    #             oldv = np.copy(self.v)
    #             # print(error,old_error)
    #             # exit()
    #             # print(abs(error-old_error))def cross_validation_als(self):
    #
    #             # old_error = np.copy(error)
    #             # print(( 2 * self.lam * np.sum(2*self.u,0).T).shape)
    #             i = np.random.permutation(self.train)[0]
    #             # print(type(i[0]))
    #             delv = 2 * self.u[int(i[0])-1].T * (self.u[int(i[0])-1]@self.v[int(i[1])-1].T-i[2]) + 2 * self.lam * self.v[int(i[1])-1]
    #             delu = 2 * self.v[int(i[1])-1] * (self.u[int(i[0])-1]@self.v[int(i[1])-1].T-i[2]) + 2 * self.lam * self.u[int(i[0])-1].T
    #             # print(delu.shape)
    #             # print(np.linalg.norm(self.u),np.linalg.norm(self.v))
    #             # print(delu)
    #             self.u[int(i[0])-1] =- self.alpha * delu
    #             self.v[int(i[1])-1] =- self.alpha * delv
    #             # print(v)
    #             error = self.error_scoring(2,self.u,self.v)
    #             # print(error)
    #             normu = np.linalg.norm(self.u[int(i[0])-1]-oldu[int(i[0])-1])
    #             normv = np.linalg.norm(self.v[int(i[1])-1]-oldv[int(i[1])-1])
    #             print(normu, normv)
    #             errors.append(error)
    #     plt.plot(errors)
    #     plt.savefig("Error.png")

    def cross_validation_als(self):
        d_range = [2,5,10,20,50]
        old_error = np.array(100000)
        error = 0
        errorm = []
        errort = []
        for o,d in enumerate(tqdm(d_range)):
            errors = []
            print(d)
            normu = 100000
            normv = 100000
            self.u = np.random.randn(self.n,d)
            self.v = np.random.randn(self.m,d)
            # print(self.u.shape,self.v.shape)
            # exit()
            # print((self.v[int(8)-1]).shape)
            # while error > .01:
            for i in tqdm(range(1000))  :
                oldu = np.copy(self.u)
                oldv = np.copy(self.v)
                for v in tqdm(range(self.v.shape[0])):

                    index = np.where(self.train[:,1].astype(int)==v+1)[0].tolist()
                    # print(index)
                    uindex = (self.train[index,0].astype(int)-1)
                    # print(uindex)
                    A = np.dot(self.u[uindex].T,self.u[uindex]) + self.lam[o] * np.eye(d)
                    B = np.dot(self.u[uindex].T,self.train[index,2])
                    vj = np.linalg.solve(A,B)
                    self.v[v] = vj

                for u in tqdm(range(self.u.shape[0])):

                    index = np.where(self.train[:,0].astype(int)==u+1)[0].tolist()
                    vindex = (self.train[index,1].astype(int)-1)
                    A = np.dot(self.v[vindex].T,self.v[vindex]) + self.lam[o] * np.eye(d)
                    B = np.dot(self.v[vindex].T,self.train[index,2])
                    uj = np.linalg.solve(A,B)
                    self.u[u] = uj

                error = self.error_scoring(2,self.u,self.v)
                # print(error)
                errors.append(error)
                # if i>2:
                #     if (abs(errors[-1]-errors[-2])/errors[-2]) < .0015 :
                #         break

                # normu = np.linalg.norm(self.u[int(i[0])-1]-oldu[int(i[0])-1])
                # normv = np.linalg.norm(self.v[int(i[1])-1]-oldv[int(i[1])-1])
                # print(normu, normv)
                # errors.append(error)
            # print(errors)
            errorm.append(errors[-1])
            errort.append(self.error_scoring(0,self.u,self.v))
            # np.save("ONE.npy",np.array(errors))
        print(errorm)
        np.save("Train.npy",np.array(errorm))
        np.save("Test.npy",np.array(errort))

    # def cross_validation_als_1(self):
    #     d_range = [1]
    #     old_error = np.array(100000)
    #     error = 0
    #     errorm = []
    #     errort = []
    #     for o,d in enumerate(tqdm(d_range)):
    #         errors = []
    #         print(d)
    #         normu = 100000
    #         normv = 100000
    #         # self.u = np.random.randn(self.n,d)
    #         self.u = np.ones((n,1))
    #         self.v = np.random.randn(self.m,d)
    #         # print(self.u.shape,self.v.shape)
    #         # exit()
    #         # print((self.v[int(8)-1]).shape)
    #         # while error > .01:
    #         for i in tqdm(range(1000))  :
    #             oldu = np.copy(self.u)
    #             oldv = np.copy(self.v)
    #             for v in tqdm(range(self.v.shape[0])):
    #
    #                 index = np.where(self.train[:,1].astype(int)==v+1)[0].tolist()
    #                 # print(index)
    #                 uindex = (self.train[index,0].astype(int)-1)
    #                 # print(uindex)
    #                 A = np.dot(self.u[uindex].T,self.u[uindex]) + self.lam[o] * np.eye(d)
    #                 B = np.dot(self.u[uindex].T,self.train[index,2])
    #                 vj = np.linalg.solve(A,B)
    #                 self.v[v] = vj
    #
    #             # for u in tqdm(range(self.u.shape[0])):
    #             #
    #             #     index = np.where(self.train[:,0].astype(int)==u+1)[0].tolist()
    #             #     vindex = (self.train[index,1].astype(int)-1)
    #             #     A = np.dot(self.v[vindex].T,self.v[vindex]) + self.lam[o] * np.eye(d)
    #             #     B = np.dot(self.v[vindex].T,self.train[index,2])
    #             #     uj = np.linalg.solve(A,B)
    #             #     self.u[u] = uj
    #
    #             error = self.error_scoring(2,self.u,self.v)
    #             # print(error)
    #             errors.append(error)
    #             # if i>2:
    #             #     if (abs(errors[-1]-errors[-2])/errors[-2]) < .0015 :
    #             #         break
    #
    #             # normu = np.linalg.norm(self.u[int(i[0])-1]-oldu[int(i[0])-1])
    #             # normv = np.linalg.norm(self.v[int(i[1])-1]-oldv[int(i[1])-1])
    #             # print(normu, normv)
    #             # errors.append(error)
    #         # print(errors)
    #         errorm.append(errors[-1])
    #         errort.append(self.error_scoring(0,self.u,self.v))
    #         # np.save("ONE.npy",np.array(errors))
    #     print(errorm)
    #     np.save("Train.npy",np.array(errorm))
    #     np.save("Test.npy",np.array(errort))





def SVD(train,trainl,test):
    d_range = [1,2,5,10,20,50]
    errors = []
    errorst = []
    for d in d_range: #TODO: Demean
        a, b,c = svds(train,d)
        # print(a.shape, b.shape, c.shape)
        tr = a@np.diag(b)@c
        print(tr.shape)
        # error_train = np.sum(np.square(np.abs(tr-train)))/np.count_nonzero(train)
        error_test = 0
        error_train = 0
        for i in trainl:
            # print(type(i[2]),type(tr[1,2]))
            error_train += np.square(np.abs(i[2] - tr[int(i[0])-1, int(i[1]) - 1]))
        error_train /= test.shape[0]
        for i in test:
            # print(type(i[2]),type(tr[1,2]))
            error_test += np.square(np.abs(i[2] - tr[int(i[0])-1, int(i[1]) - 1]))
        error_test /= test.shape[0]
        # print(error_train.shape)
        # exit()
        errors.append(error_train)
        errorst.append(error_test)
    plt.plot(d_range,errors,"-.", label = "Training")
    plt.plot(d_range,errorst,"-.", label = "Testing" )
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig("SVD.png")


def Q5(train,trainl,test):
    print(train.shape,trainl.shape,test.shape)
    u = np.ones((1000,1))
    v = np.nanmean(train,axis = 1)
    print(v.shape)
