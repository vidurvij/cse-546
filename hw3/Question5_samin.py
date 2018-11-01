import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from tqdm import tqdm

def load_dataset():
    mndata = MNIST()
    mndata.gz = True
    x_train, labels_train = map(np.array, mndata.load_training())
    x_test, labels_test = map(np.array, mndata.load_testing())
    x_train = x_train/255.0
    x_test = x_test/255.0
    return x_train, labels_train, x_test, labels_test


X_train, label_train, X_test, label_test = load_dataset()

label_train = label_train.reshape(label_train.shape[0])
label_test = label_test.reshape(label_test.shape[0])

print('X_train', X_train.shape, 'label_train', label_train.shape)


index_train =np.where(np.logical_or(label_train==2, label_train==7))
index_test = np.where(np.logical_or(label_test==2, label_test==7))


label_train = label_train[index_train]
label_test = label_test[index_test]
X_train = X_train[index_train]
X_test = X_test[index_test]

# print(label_train[0:20])

label_train = np.where(label_train==7, 1, -1)
label_test = np.where(label_test==7, 1, -1)

# print(label_train[0:20])

print('label_train', label_train.shape, 'X_train', X_train.shape, 'label_test', label_test.shape, 'X_test', X_test.shape)

mu = []
def mu_func(z):

    mu = 1.0 / (1.0 + np.exp(-z))
    return mu


lamda = 0.1
W = np.zeros((X_train.shape[1], 1))
b = 0


def computation(X, Y, W, b):

    # mu = np.reciprocal(1 + np.exp(-(Y*(b + X @ W))))
    # print('mu', mu)
    # print('etha_shape', etha.shape)
    # logistic_loss = np.log(1/mu)
    # print("_---------",Y.shape,X.shape,W.shape)
    z = np.multiply(Y.reshape(X.shape[0], 1), (b + X @ W))
    # print('z_value', z)
    mu = mu_func(z)
    # print('mu_value', mu.shape)
    mu = np.reshape(mu, (mu.shape[0], 1))
    # print('mu_shape', mu.shape)
    J = np.mean(np.log(np.reciprocal(mu))) + lamda*(W.T @ W)
    #print("^^^^^^^",(np.mean(-np.multiply((np.multiply(X, Y.reshape(Y.shape[0],1))), (1-mu)), axis=0)).shape,(2*lamda*W).shape)
    grad_W = (np.mean(-np.multiply((np.multiply(X, Y.reshape(Y.shape[0],1))), (1-mu)), axis=0)).reshape(X.shape[1],1) + (2*lamda*W)
    #print ("-----",grad_W.shape)

    # print('grad_W_shape', grad_W.shape)
    # grad_W = np.reshape(grad_W, )
    # print('grad_W_shape', grad_W.shape)
    grad_b = np.mean(-np.multiply(Y, (1-mu)))
    # print('grad_b_shape', grad_b.shape)
    # print('grad_b', grad_b.shape)
    return J, grad_W, grad_b


def optimization(X, Y, W, b, num_iteration, step_size):
    J_list = []
    W_list = []
    for i in tqdm(range(num_iteration)):
        J, grad_W, grad_b = computation(X, Y, W, b)
        # print('cost', J)
        W = W - step_size*grad_W
        b = b - step_size*grad_b
        J_list = np.append(J_list, J)
        W_list = np.append(W_list, W)
    return J_list, W_list


def sign(X, W, b):
    sgn = np.sign(b + X @ W)
    return sgn


def misclass_error(X, Y, W, b):
    misclass = []
    sgn = sign(X, W, b)
    misclass = (np.nonzero(Y - sgn))/Y.shape[0]

    return misclass


cost_train, W = optimization(X_train, label_train, W, b, 200, 0.2)
# cost_test, W = optimization(X_test, label_test, W, b, 10, 0.0001)


# print('cost_train', cost_train)
# print('cost_test', cost_test)
# print('W', W)

# plt.plot(np.linalg.norm(W))
plt.plot(cost_train)
plt.show()

exit()

# y_train_binary = (label_train == 5).astype(np.int)
# y_test_binary = (label_test == 5).astype(np.int)
#
# print(label_train[0:20], y_train_binary[0:20])
