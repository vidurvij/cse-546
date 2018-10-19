from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pa
classes = 10

def load_data():
    mndata = MNIST()
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, X_test, labels_test, labels_train

def onehot(labels, clas):
    onehots = []
    for label in labels:
        onehot = np.zeros((clas,label.shape[0]))
        for i,pos in enumerate(label):
            onehot[pos][i] = 1
        onehots.append(onehot)
    return onehots

def train(Y,X,l):
    #print('y --',Y.shape)
    #print('x--',X.shape)
    #w = Y@X.T@np.linalg.inv(X@X.T-l*np.identity(X.shape[0]))
    w = np.linalg.solve(X.T@X-l*np.identity(X.shape[1]),X.T@Y.T)
    #print("w--",w.shape)
    return w

def predict(X,W):
    p = X@W
    #print("^^^^",p.shape)
    result = np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
        result[i] = np.argmax(p[i,:])
    #print (result.shape)
    return result

def metric(y,label):
    res = np.sum(np.equal(y,label.reshape(label.shape[0],1)))
    metric = ((res)/label.shape[0])*100
    #print("The given model was "+str(metric)+"% accurate for given dataset")
    return metric

def random_split(x,y,split):
    assert split<1
    assert x.shape[0]==y.shape[0]
    ids = np.random.permutation(x.shape[0])
    x, y = x[ids], y[ids]
    #print("1:",x)
    #print (np.rint(split*x.shape[0]))
    #return x[0:(np.rint(split*x.shape[0]),:]).astype(np.intp),y[0:(np.rint(split*x.shape[0])).astype(np.intp),:],x[(np.rint(split*x.shape[0])).astype(np.intp):x.shape[0],:],y[(np.rint(split*x.shape[0])).astype(np.intp):x.shape[0],:]
    return x[0:int(x.shape[0]*split),:],y[0:int(x.shape[0]*split)],x[int(x.shape[0]*split):x.shape[0],:],y[int(x.shape[0]*split):x.shape[0]]

def feature_transform(xs,p):
    xr = []
    #print (xs[0].shape)
    g = np.sqrt(.1)*np.random.randn(xs[0].shape[1],p)
    b = np.random.uniform(0,2*np.pi,(1,p))
    for x in xs:
        x = (x@g)+b
        x = np.cos(x)
        xr.append(x)
        #print(g.shape)
    return xr
def plotter(data):
    a = sns.relplot(kind = "line",data=pa.DataFrame(data))
    a.set_xlabels('Feature Vector Length P')
    a.set_ylabels('Accuracy of the Model')
    a.savefig('Final_Question.png')
    plt.show()

def model_validation(train_data, train_label, valid_data, valid_label):
    result1,result2 = [], []
    labels = onehot([train_label,valid_label,],classes)
    # print(train_data.shape,valid_data.shape)
    valid_old = 10000
    train_old = 10000
    index = 0
    for p in tqdm(range(1500)):
        #print(train_data.shape)
        data = feature_transform([train_data,valid_data],p+1)
        #print ('#',train_data_t.shape)
        #valid_data_t = feature_transform(valid_data.T,p+2)
        W = train(labels[0],data[0],10^-4)
        predict_train = predict(data[0],W)
        predict_val = predict(data[1],W)
        acc_train = metric(predict_train,train_label)
        acc_valid = metric(predict_val,valid_label)
        result1.append(acc_train)
        result2.append(acc_valid)
        index  = p
        if abs(acc_valid-valid_old)<0.1:
            break
        # valid_old = acc_valid

    result1 = np.array(result1)
    result2 = np.array(result2)
    plotter({"Training Accuracy":result1,"Validation Accuracy":result2})
    #print ("##########################",result.shape)
    # plt.plot(result1)
    # plt.plot(result2)
    # plt.show()
    np.save('result1.npy',result1)
    np.save('result1.npy',result2)
    return index

def model_test(X_train, X_test, labels_test, labels_train,p):
    delta = .05
    labels = onehot([labels_train,labels_test],classes)
    data = feature_transform([X_train,X_test],p)
    w = train(labels[0],data[0],10^-4)
    test_predict = predict(data[1],w)
    E_test = (metric(test_predict,labels_test)/100)
    epsilon = np.sqrt((np.log(2/delta))/(2*labels_test.shape[0]))
    lower_c = E_test - epsilon
    upper_c = E_test + epsilon
    print("The confidence limits are from "+str(lower_c)+" to "+str(upper_c)+" around the test accuracy of "+str(E_test))
    file = open("Log.Txt",'w')
    file.write("The confidence limits are from "+str(lower_c)+" to "+str(upper_c)+" around the test accuracy of "+str(E_test))
    file.close


X_train, X_test, labels_test, labels_train = load_data()
#print (X_train.shape)-
labels = onehot([labels_train,labels_test],classes)
w = train(labels[0],X_train,10^-4)
result = predict(X_train,w)
print("Training Data metric: ",metric(result,labels_train))
result = predict(X_test,w)
print("Testing Data metric: ",metric(result,labels_test))
train_data, train_label, valid_data, valid_label = random_split(X_train,labels_train,.8)
# print("2:",train_data)
# print("3:",valid_data)
index = model_validation(train_data, train_label, valid_data, valid_label)
print("Index: ", index)
model_test(X_train, X_test, labels_test, labels_train,index)
