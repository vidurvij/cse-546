#Question 3

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
        result_array = np.isin(np.nonzero(w)[0],np.nonzero(wo)[0])
        fdr = (np.count_nonzero(w)-np.sum(result_array))/np.count_nonzero(w)
        tpr = np.sum(result_array)/np.count_nonzero(wo)
        return fdr,tpr


def random_data(): # Generating Random Data with n features and d examples
    x = np.random.randn(n,d)
    #x = preprocessing(x)
    w = np.zeros((d,1))
    for i in range(k):
        w[i] = (i+1)/k
    print (w)
    question_one_metrics(w)
    #print (w)
    y = w.T@x.T + np.random.randn(1,n)
    print (y.shape)
    #print (y.shape,x.shape)
    return x,y

def model_definiton(x,y):
    lambda_max = np.max(2*np.absolute(x.T@(y-np.mean(y)).T))
    print (lambda_max)
    return lambda_max

def plotter(x , ys , title, xlabel, ylabel, flag,flag2 = False, flag3 = False):
    if not (os.path.exists(os.path.basename(__main__.__file__[:-3]))):
        os.mkdir(os.path.basename(__main__.__file__[:-3]))
    plt.clf()
    for y in ys:
        plt.plot(x,y)
    plt.plot(x,y)
    if flag:
        plt.gca().invert_xaxis()
    if flag2:
        plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if flag3:
        plt.legend(["Training error","Validation Error"])
    #plt.show()
    plt.savefig(os.path.basename(__main__.__file__[:-3])+"/"+title+".png")

def lasso(x,y,lam):
    print(y.shape)
    w = np.zeros((d,1))
    fdrs = []
    tprs = []
    sums = []
    changes = []
    lams = []
    delta = 100000
    wold = np.repeat(100,d).reshape(d,1)
    wold2 = np.repeat(100,d).reshape(d,1)
    print(x.shape)
    # while delta > .0001:
    for j in tqdm(range(500)):
        wold = np.copy(w)
        wold2 = np.repeat(100,d).reshape(d,1)
        while (np.max(np.abs(w-wold2))) > .01: #TODO: Optimize the stopping criteria
            #print(np.max(np.abs(w-wold)))
            #print("Hit")
            wold2 = np.copy(w)
            b = np.mean(y-w.T@x)
            for k in range(x.shape[0]):
                wj = np.copy(w)
                wj[k] = 0
                # print(y.shape,wj.shape,x.shape)
                # exit()
                a = 2*(np.sum(np.square(x[k])))
                #print(x[k].shape,y.shape,wj.shape)
                c = 2*(x[k].reshape(1,x.shape[1])@(y-(b+wj.T@x)).T) #TODO check
                if c < -lam:
                    wk = (c+lam)/a
                if c >= -lam and c <= lam:
                    wk = 0
                if c > lam:
                    wk = (c-lam)/a
                #print (c, lam, wk)
                w[k] = wk
                b = np.mean(y-w.T@x)
            sum = np.sum(np.square(w.T@x-y+b))+lam*np.linalg.norm(w,1)
            #print(np.max(np.abs(w-wold)))
                #print(sum)
            #print (w[0:100])
            #sum = np.sum(np.square(w.T@x-y+b))+lam*np.linalg.norm(w,1)
            #print (w)
            #print(i,w,wold)
            sums.append(sum)

            #print (change)
        fdr, tpr = question_one_metrics(w,False)
        fdrs.append(fdr)
        tprs.append(tpr)
        change = np.count_nonzero(w)
        changes.append(change)
        lams.append(lam)
        lam = lam/1.5
    #print(w)
    return changes, fdrs, tprs, lams, sums

def Question_3():
    x,y = random_data()
    lam = model_definiton(x,y)
    print("Lam:",lam)
    changes, fdrs, tprs, lams, sums = lasso(x.T,y,lam)
    print (len(lams[1:]))
    plotter(lams[1:],[changes[1:]],title = "Number of features vs Lambda ", xlabel = "Lambda", ylabel = "Number of Features",flag = True, flag2 = True, flag3 = True)
    plotter(fdrs[1:],[tprs[1:]],title = "Fdrs vs tprs ", xlabel = "FDRS", ylabel = "TPRS ",flag = False)
    plotter(np.arange(1,len(sums)+1), [sums], title = "Cost vs Iterations", xlabel = "Iterations", ylabel = "Cost", flag = False)

if __name__ == "__main__":
    Question_3()
