#Question 4
from Question3 import *
from tqdm import tqdm
#import matplotlib.pylplot as plt



def data_split(x,y):
    return x[0:4000], x[4000:5000], x[5000:6000], y[0:4000], y[4000:5000], y[5000:6000]

# Load a csv of floats:
X = np.genfromtxt("upvote_data.csv", delimiter=",")
# Load a text file of integers:
y = np.loadtxt("upvote_labels.txt", dtype=np.int)
# Load a text file of strings:
featureNames = open("upvote_features.txt").read().splitlines()
y = np.sqrt(y)
x_train, x_valid, x_test, y_train, y_valid, y_test = data_split(X,y)

def predict(x,w):
    return x.T@w



def compute_error(y_predict,y_valid):
    error = np.mean(np.square(y_predict-y_valid))
    return error


def lasso(x ,y,x_valid,y_valid,lam):
    errors_valid = []
    errors_train = []
    lams = []
    sums = []
    changes = []
    ws = []
    # while delta > .0001:
    w = w = np.zeros((x.shape[0],1))
    for j in tqdm(range(50)):
        wold = np.copy(w)
        wold2 = np.repeat(100,d).reshape(d,1)
        iter = 0
        while (np.max(np.abs(w-wold2))) > .1 and iter <= 20: #TODO: Optimize the stopping criteria
        #for i in range(1):
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
            iter +=1
        change = np.count_nonzero(w)
        y_predict_valid = predict(x_valid,w)
        y_predict_train = predict(x,w)
        error_train = compute_error(y_predict_train,y)
        error_valid = compute_error(y_predict_valid,y_valid)
        changes.append(change)
        errors_train.append(error_train)
        errors_valid.append(error_valid)
        lams.append(lam)
        sums.append(sum)
        lam = lam/1.5
        ws.append(w)
    return errors_valid,errors_train, lams, changes, np.array(ws), sums


lam = model_definiton(x_train,y_train)
errors_valid,errors_train, lams,changes,w, sums = lasso(x_train.T,y_train.T,x_valid.T,y_valid.T,lam)
np.save("Weight.npy",w)
# w_sorted = np.argsort(w, "mergesort")
# print(w_sorted)
# print(w)
plotter(lams,[errors_train,errors_valid],"Lambda vs Error","Lambda","Error", True, True,True)
plotter(lams,[changes],"Lambda vs Change","Lambda","Number of Features in consideration", True,True)
plotter(np.arange(0,len(changes)),[sums],"Cost vs Iterations", "Iterations","Cost", False)

errors = []
w_min = w[np.argmin(errors_valid)]
print(lams[np.argmin(errors_valid)])
for x,y in ([(x_test,y_test),(x_train,y_train),(x_valid,y_valid)]):
    y_predict = predict(x.T,w_min)
    error = compute_error(y,y_predict)
    errors.append(error)

print("The Test, Train and Validation error on the trained model is  " + str(errors))
w_sorted = np.argsort(w_min,axis=0)[0:10]

# for i in range(w_sorted.shape[0]):
#     w_sorted[i] = np.asscalar(w_sorted[i])
# #print(np.apply_along_axis(np.asscalar,0,w_sorted))
# print(type(np.asscalar(w_sorted[1])))
for i in range(w_sorted.shape[0]):
    print("Feature "+str(i)+":"+featureNames[np.asscalar(w_sorted[i])])
print("The ideal lamda is:",lams[np.argmin(errors_valid)])
