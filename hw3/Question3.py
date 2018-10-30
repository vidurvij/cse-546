import numpy as np
import scipy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import jit
import os
import __main__


n = 500
d = 1000
k = 100
wo = np.array(0)

def preprocessing(x):
    x_mean = np.mean(x,axis = 1)
    x = x-x_mean.reshape(500,1)
    #print (np.mean(x,axis = 1))

    return x

def question_one_metrics(w,initial = True):
    if initial:
        global wo
        wo = w.copy()
    if not (initial):
        result_array = np.isin(np.nonzero(w),np.nonzero(wo))
        # print(np.sum(result_array))
        fdr = np.sum(np.invert(result_array))/np.count_nonzero(w)
        #print(np.sum(np.equal(wo,w)))
        tpr = np.sum(result_array)/np.count_nonzero(wo)
        return fdr,tpr


def random_data(): # Generating Random Data with n features and d examples
    x = np.random.randn(500,1000)
    #x = preprocessing(x)
    w = np.zeros((500,1))
    for i in range(k):
        w[i] = (i+1)/k
    print (w)
    question_one_metrics(w)
    #print (w)
    y = w.T@x + np.random.randn(1,1000)
    #print (y.shape,x.shape)
    return x,y

def model_definiton(x,y):
    lambda_max = np.max(2*np.absolute(x@(y-np.mean(y)).T))
    print (lambda_max)
    return lambda_max

def plotter(x , ys , title, xlabel, ylabel, flag):
    if not (os.path.exists(os.path.basename(__main__.__file__[:-3]))):
        os.mkdir(os.path.basename(__main__.__file__[:-3]))
    plt.clf()
    for y in ys:
        plt.plot(x,y)
    plt.plot(x,y)
    if flag:
        plt.gca().invert_xaxis()
    # plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.savefig(os.path.basename(__main__.__file__[:-3])+"/"+title+".png")

def lasso(x,y,lam):
    w = np.zeros((500,1))
    fdrs = []
    tprs = []
    changes = []
    lams = []
    delta = 100000
    wold = np.repeat(100,500).reshape(500,1)
    wold2 = np.repeat(100,500).reshape(500,1)
    # while delta > .0001:
    for i in tqdm(range(500)): #TODO: Optimize the stopping criteria
        wold = np.copy(w)
        b = np.mean(y-w.T@x)
        for k in range(x.shape[0]):
            wj = np.copy(w)
            wj[k] = 0
            # print(y.shape,wj.shape,x.shape)
            # exit()
            a = 2*(np.sum(np.square(x[k])))
            #print(x[k].shape,y.shape,wj.shape)
            c = 2*(x[k].reshape(1,1000)@(y-(b+wj.T@x)).T) #TODO check
            if c < -lam:
                wk = (c+lam)/a
            if c >= -lam and c <= lam:
                wk = 0
            if c > lam:
                wk = (c-lam)/a
            #print (c, lam, wk)
            w[k] = wk
        sum = np.sum(np.square(w.T@x-y+b))+lam*np.linalg.norm(w,1)
        #print(np.max(np.abs(w-wold)))
            #print(sum)
        #print (w[0:100])
        #sum = np.sum(np.square(w.T@x-y+b))+lam*np.linalg.norm(w,1)
        lams.append(lam)
        lam = lam/1.01
        #print (w)
        #print(i,w,wold)
        change = np.count_nonzero(w-wold)
        fdr, tpr = question_one_metrics(w,False)
        fdrs.append(fdr)
        tprs.append(tpr)
        #print (change)
        changes.append(change)
    #print(w)
    return changes, fdrs, tprs, lams

def Question_3():
    x,y = random_data()
    lam = model_definiton(x,y)
    changes, fdrs, tprs, lams = lasso(x,y,lam)
    print (len(lams[1:]))
    plotter(lams[1:],[changes[1:]],title = "Number of features vs Lambda ", xlabel = "Lambda", ylabel = "Number of Features",flag = True)
    plotter(fdrs[1:],[tprs[1:]],title = "Fdrs vs tprs ", xlabel = "FDRS", ylabel = "TPRS ",flag = True)

if __name__ == "__main__":
    Question_3()
