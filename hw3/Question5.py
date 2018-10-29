import numpy as np
from mnist import MNIST
from Question3 import *




lam = .1
iterations = 1000
def load_data():
    mndata = MNIST()
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    index = np.where(np.logical_or((labels_test == 2),(labels_test == 7)))
    labels_test = labels_test[index]
    # print(labels_test.shape)
    labels_test = np.where(labels_test == 7, 1,-1)
    # print(labels_test.shape)
    X_test = X_test[index]
    index2 = np.where(np.logical_or((labels_train == 2),(labels_train == 7)))
    labels_train = labels_train[index2]
    # print(labels_train.shape)
    labels_train = np.where(labels_train == 7, 1,-1)
    # print(labels_train.shape)
    X_train = X_train[index2]
    # print(np.where(np.logical_or((labels_test == 2),(labels_test == 7))))
    # print(X_train.shape)
    return X_train, X_test, labels_train, labels_test

def calculate_cost(x,w,y,b):
    # print(x.shape,y.shape,w.shape,b.shape)
    # print(type(np.mean(np.log(1+np.exp(-y*(b+x@w)))) + lam*np.sum(np.square(w))))
    # print(-y*(b+x@w))
    # exit()
    return np.mean(np.log(1+np.exp(-y*(b+x@w)))) + lam*np.sum(np.square(w))

def predict(w,b,x):
    # print("-------",w.shape,x.shape)
    return np.sign(np.exp(b+x@w))

def compute_error(y_predict,y):
    return np.sum(np.invert(np.equal(y_predict,y)))/y.shape[0]

def gradient_descent(x,y,xt,yt,stochastic,batch = 1):
    w = np.zeros((x.shape[1],1))
    b = 0
    print("Sixe:",x.shape,y.shape)
    ws = []
    bs = []
    costs = []
    costst = []
    errors = []
    errorst = []
    costs.append(np.asscalar(calculate_cost(x,w,y,b)))
    costst.append(np.asscalar(calculate_cost(xt,w,yt,b)))
    errors.append(compute_error(predict(w,b,x),y))
    errorst.append(compute_error(predict(w,b,xt),yt))
    ws.append(np.linalg.norm(w))
    bs.append(np.linalg.norm(b))
    for i in tqdm(range(iterations)):

        #print("-----------",mu.shape,y.shape,(np.mean((1-mu).T*(-y.T*x.T),axis = 1).reshape(x.shape[1],1) ).shape)
        #print((np.mean((1-mu)*(-y*x),axis = 0) ).shape)
        if stochastic:
            ids = np.random.permutation(x.shape[0]) # TODO: Check whether this is right or not
            x_rand = x[ids]
            y_rand = y[ids]
            mu = np.reciprocal(1+np.exp(-y_rand[0:batch]*(b+x_rand[0:batch]@w)))
            delw = np.mean((1-mu).T*(-y_rand[0:batch].T*x_rand[0:batch].T),axis = 1).reshape(x.shape[1],1) + 2*(lam/iterations)*w
            delb = np.mean((1-mu)*(-y_rand[0:batch]))
            #print(delw.shape,delb.shape)
        if not stochastic:
            mu = np.reciprocal(1+np.exp(-y*(b+x@w)))
            delw = np.mean((1-mu).T*(-y.T*x.T),axis = 1).reshape(x.shape[1],1) + 2*lam*w
            delb = np.mean((1-mu)*(-y))
        w = w - .9*delw
        #print("##########:",w.shape)
        b = b - .5*delb
        cost = np.asscalar(calculate_cost(x,w,y,b))
        costt = np.asscalar(calculate_cost(xt,w,yt,b))
        p = predict(w,b,x)
        pt = predict(w,b,xt)
        error = compute_error(p,y)
        errort = compute_error(pt,yt)
        errors.append(error)
        errorst.append(errort)
        #print(cost)
        ws.append(np.linalg.norm(w))
        bs.append(np.linalg.norm(b))
        costs.append(cost)
        costst.append(costt)

    print(len(errors))
    print(len(ws))
    print(type(np.arange(0,iterations+1).tolist()))
    plotter(np.arange(0,iterations+1).tolist(),[costs],title = "Cost_Iteration", xlabel = "Iterations", ylabel = "Cost",flag = False)
    plotter(np.arange(0,iterations+1).tolist(),[costst],title = "Cost_Iteration_test", xlabel = "Iterations", ylabel = "Cost",flag = False)
    plotter(np.arange(0,iterations+1).tolist(),[ws],title = "W_Iterations", xlabel = "Iterations", ylabel = "W",flag = False)
    plotter(np.arange(0,iterations+1).tolist(),[bs],title = "b_Iterations", xlabel = "Iterations", ylabel = "b",flag = False)
    plotter(np.arange(0,iterations+1).tolist(),[errors,errorst],title = "Missclassification Error", xlabel = "Iterations", ylabel = "Error",flag = False)
    return w,b


x_train, x_test, y_train, y_test = load_data()
# print(y_train,y_test)
# print(x_train.shape,y_train.shape)
gradient_descent(x_train,y_train.reshape(y_train.shape[0],1),x_test,y_test.reshape(y_test.shape[0],1),True,1000)
