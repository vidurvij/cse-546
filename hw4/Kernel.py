import numpy as np
import matplotlib.pyplot as plt
class Kernel():
    def __init__(self,train,label, kernel, p) :
        self.train = train
        self.label = label
        self.kernel = kernel
        # self.K = 0
        self.lam = .000001
        self.alpha = 0
        self.parameter = p
    def Regression(self,train,label):
        # print(K.shape)
        train = train.reshape(train.shape[0],1)
        K = self.Kernel(train,train)
        # print(train.shape)
        tem = K+self.lam*np.eye(K.shape[0])
        alpha = np.linalg.solve(tem,label)
        return alpha
        # print(alpha.shape)
        # error += self.error_compute(self.train[i], train, self.label[i],alpha

    def error_compute(self,x1,x2, y, alpha = 0):
        if isinstance(alpha, int):
            alpha = self.alpha
        K = self.Kernel(x1,x2)
        y_hat = np.sum(alpha * K)
        error = (y_hat-y)**2
        return error

    def predict(self,x1,x2, alpha):
        K = self.Kernel(x1,x2)
        y_hat = np.sum(alpha * K)
        return y_hat

    def curve(self,train, alpha,lam):
        y = []
        xs = []
        for x in np.arange(0.0, 1.0, 0.01):
            # print(train.shape)
            y.append(self.predict(np.array(x).reshape(1,1),train, alpha))
            xs.append(x)
        plt.plot(xs,y,label = str(lam))

    def cross_validation(self, parameter):
        errors = []
        lams = []
        lam = parameter
        for i in range(10):
            error = 0
            for i in range(self.train.shape[0]):
                train = np.delete(self.train,i).reshape(self.train.shape[0]-1,1)
                # print(train.shape)
                label = np.delete(self.label,i)
                # print(trai1n.shape)
                K = self.Kernel(train,train)
                tem = K+lam*np.eye(K.shape[0])
                alpha = np.linalg.solve(tem,label)
                # print(alpha.shape)
                error += self.error_compute(self.train[i], train, self.label[i],alpha)
            self.curve(train,alpha,lam)
            lams.append(lam)
            lam *= 1.5
            errors.append(error/(train.shape[0]-1))
        plt.ylim(-10,10)
        self.plot_true()
        plt.legend()
        plt.show()
        plt.clf()
        plt.ylim(0,10)
        plt.plot(lams ,errors)
        # plt.gca().invert_xaxis()
        plt.xscale('log')
        plt.show()
        # self.lam = np.array(lams)[argmin(np.array(errors))]
    def plot_true(self):
        x = np.arange(0,1,.01)
        # print(x)
        y =4 * np.sin(np.pi*x) * np.cos(6*np.pi*(x**2))
        # print(y)1
        plt.plot(x,y)
        # plt.show()
    def Kernel(self,x1,x2):
        # K = numpy.zeros((train.shape[0],train.shape[0]))
        if self.kernel == 1:
            return (1 + x1 @ x2.T)**self.parameter
        if self.kernel == 2:
            # print("!!!",x2.shape)
            K = np.zeros((x1.shape[0],x2.shape[0]))
            for i in range(x1.shape[0]):
                temp = x1[i] - x2.T
                K[i,:] = np.sum(temp*temp, axis = 0)
            K = np.exp(-self.parameter*K)
            # print(K.shape)
            return K

    def Bootstrap(self,b):
        train = np.zeros((b,self.train.shape[0]))
        labels = np.zeros((b,self.train.shape[0]))
        alpha = np.zeros((b,self.train.shape[0]))
        ys = np.zeros((b,self.train.shape[0]))
        for i in range(b) :
            index = np.random.choice(self.train.shape[0], self.train.shape[0])
            train[i] = self.train[index.tolist()].T
            labels[i] = self.label[index]
            alpha[i] = self.Regression(train[i],labels[i])
            for j in range(self.train.shape[0]):
                # print(j)
                # self.train[j], train[i].reshape(train[i].shape[0],1), alpha[i]
                ys[i][j] = self.predict(self.train[j], train[i].reshape(train[i].shape[0],1), alpha[i])
        self.confidence(ys,5)

    def confidence(self,ys,c):
            for i in range(ys.shape[1]):
                ys[:,i] = np.sort(ys[:,i])
            print(ys.shape)
            br = int(((100-c)/2)*ys.shape[0]/100)
            print(br)
            top = ys[-br:,:]
            bottom = ys[0:br,:]
            print(top.shape, bottom.shape)
            # plt.clf
            print(self.train.shape)
            self.plot_true()
            plt.scatter(self.train.T.repeat(br,0),top,marker = ".",s = 1)
            plt.scatter(self.train.T.repeat(br,0),bottom,marker = ".", s = 1)
            plt.ylim(-10,10)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("n = 30, $\gamma$ = " +str(self.lam)+", d ="+str(self.parameter) )
            # plt.show()
            plt.savefig("Kernel/BootstrapKernel2-300-c-"+str(c)+".png")
    # def Kernel_2(self,x,d):
        # for i in range
        # K = np.zeros((self.train.shape[0], self.train.shape[0]))
