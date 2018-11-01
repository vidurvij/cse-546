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
    # while delta > .0001:
    w = w = np.zeros((x.shape[0],1))
    for j in tqdm(range(500)):
        wold = np.copy(w)
        for i in tqdm(range(1)): #TODO: Optimize the stopping criteria
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
        change = np.count_nonzero(w-wold)
        y_predict_valid = predict(x_valid,w)
        y_predict_train = predict(x,w)
        error_train = compute_error(y_predict_train,y)
        error_valid = compute_error(y_predict_valid,y_valid)
        changes.append(change)
        errors_train.append(error_train)
        errors_valid.append(error_valid)
        lams.append(lam)
        sums.append(sum)
        lam = lam/1.01
    return errors_valid,errors_train, lams, changes, w, sums


lam = model_definiton(x_train,y_train)
errors_valid,errors_train, lams,changes,w, sums = lasso(x_train.T,y_train.T,x_valid.T,y_valid.T,lam)
plotter(lams,[errors_train,errors_valid],"Lambda vs Error","Lambda","Error", True)
plotter(lams,[changes],"Lambda vs Change","Lambda","Number of Features in consideration", True)
plotter(np.arange(0,len(changes)),[sums],"Cost vs Iterations", "Iterations","Cost", False)

errors = []
for x,y in ([(x_test,y_test),(x_train,y_train),(x_valid,y_valid)]):
    y_predict = predict(x.T,w)
    error = compute_error(y,y_predict)
    errors.append(error)

print("The Test, Train and Validation error on the trained model is  " + str(errors))
