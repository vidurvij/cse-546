import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpy import huber

np.random.seed(546)


def gen_data(n):
    x = [(i-1)/(n-1) for i in range(1, n+1)]
    x = np.reshape(x, (n, 1))
    epsilon = np.random.normal(0, 1, n).reshape(n, 1)
    f = [np.sum(1 if x[i] >= k/5 else 0 for k in range(1, 5)) for i in range(n)]
    f = np.reshape(f, (n, 1))
    f = f*10
    y = f + epsilon
    np.put(y, [24], [0])
    return x, y, f


def loss_fn_ls(k, y, alpha):
    loss = cp.sum_squares(k*alpha - y)
    return loss


def objective_fn(k, y, alpha, lam1, lam2,  loss):

    if loss == 'ls':
        obj = cp.sum_squares(k*alpha - y) + lam1*cp.quad_form(alpha, k)

    elif loss == 'huber':
        obj = cp.sum(huber(k * alpha - y, 1)) + lam1*cp.quad_form(alpha, k)

    elif loss == 'denoise':
        obj = cp.sum_squares(k * alpha - y) + lam2*cp.norm(cp.matmul(Diff(k.shape[0]), k)*alpha, 1) + lam1*cp.quad_form(alpha, k)

    else:  #'non-decreasing'
        obj = cp.sum_squares(k * alpha - y) + lam1 * cp.quad_form(alpha, k)

    return obj


def Diff(n):
    D = np.zeros((n -1, n))
    for i in range(n-1):
        for j in range(n):
            D[i][j] = np.where(i==j, -1, (np.where(i==j-1, 1, 0)))
    return D


def kernel(x, z, gam):
    k = np.exp(-gam * (np.repeat(z, x.shape[0], axis=1) - np.repeat(x.T, z.shape[0], axis=0)) ** 2)
    return k


def train(K, lam, y):
    alpha = np.linalg.solve(K+lam*np.identity(K.shape[0]), y)
    return alpha


def predict(K, alpha):
    y_hat = np.matmul(K, alpha)
    return y_hat


def CV_Error_KFold(x, y, lam1, lam2, gam, num_fold, loss):
    CV_error_list = []
    fold_len = np.int(x.shape[0]/num_fold)
    idx = np.random.permutation(x.shape[0])
    for i in range(num_fold):
        start = i*fold_len
        end = (i + 1)*fold_len
        idx_valid = idx[start:end]
        idx_train = np.delete(idx, np.arange(start, end))
        x_train = x[idx_train]
        y_train = y[idx_train]
        x_valid = x[idx_valid]
        y_valid = y[idx_valid]
        k_train = kernel(x_train, x_train, gam)
        k_valid = kernel(x_train, x_valid, gam)

        y_train = np.asarray(y_train).reshape(-1)
        alpha = cp.Variable(k_train.shape[0])
        obj = objective_fn(k_train, y_train, alpha, lam1, lam2, loss)

        if loss == 'non_decreasing':
            constraint = [((cp.matmul(Diff(k_train.shape[0]), k_train)) * alpha) >= 0]
            cp.Problem(cp.Minimize(obj), constraint).solve()
        else:
            cp.Problem(cp.Minimize(obj)).solve()

        alpha_trn = alpha.value

    # alpha_trn = train(k_train, lam, y_train)
        f_hat = predict(k_valid, alpha_trn)
        error = (f_hat - y_valid)**2
        CV_error_list = np.append(CV_error_list, error)
    CV_error = np.mean(CV_error_list)

    return CV_error


'''choosing the best gamma and lambda for least square error and Huber and de_nois'''
'''You need to change the flag in CV_Error_KFold function: True = Huber / False = least square
    and stop condition for lambda and range for gamma'''


'''Different numbers for least square '''
# condition1 = 10e-7
# condition2 = 10e-7
# m = 60 # range for gamma
# num_fold = 50
# loss = 'ls'

'''Different numbers for Huber '''
# n = 50
# condition1 = 10e-7
# condition2 = 10e-7
# m = 200 # range for gamma
# num_fold = 50
# loss = 'huber'

'''Different numbers for de_noising '''
n = 50
condition1 = 10e-7
condition2 = 10e-7
m = 50 # range for gamma
num_fold = 50
loss = 'denoise'

'''Different numbers for non_decreasing '''
# condition1 = 10e-7
# condition2 = 10e-7
# m = 50 # range for gamma
# num_fold = 50
# loss = 'non_decreasing'


CV_error_g = []
CV_error_lam1 = []
CV_error_lam2 = []

g_list = []
lamda_list1 = []
lamda_list2 = []


x, y, f = gen_data(50)


lamda1 = 1000
lamda2 = 1000

count2 = 0
count1 = 0


if loss == 'denoise':

    while lamda2 > condition2:
        count2 += 1
        print('count2', count2)

        count1 = 0
        lamda1 = 1000

        while lamda1 > condition1:
            count1+=1
            print('count1', count1)
            iter=0

            for g in range(0, m, 10):
                iter+=1
                print('iter', iter)
                error_g = CV_Error_KFold(x, y, lamda1, lamda2, g, num_fold, loss)
                CV_error_g = np.append(CV_error_g, error_g)
                g_list = np.append(g_list, g)
            hyp_g = g_list[list(CV_error_g).index(min(CV_error_g))]
            print('hyp_g', hyp_g)
            error_lam1 = CV_Error_KFold(x, y, lamda1, lamda2,  hyp_g, num_fold, loss)
            CV_error_lam1 = np.append(CV_error_lam1, error_lam1)
            lamda_list1 = np.append(lamda_list1, lamda1)
            lamda1 = lamda1*0.1

        regul_lam1 = lamda_list1[list(CV_error_lam1).index(min(CV_error_lam1))]
        print('regul_lam1', regul_lam1)

        error_lam2 = CV_Error_KFold(x, y, regul_lam1, lamda2, hyp_g, num_fold, loss)
        print('error_lam2', error_lam2)
        CV_error_lam2 = np.append(CV_error_lam2, error_lam2)
        lamda_list2 = np.append(lamda_list2, lamda2)
        lamda2 = lamda2 * 0.1

    regul_lam2 = lamda_list2[list(CV_error_lam2).index(min(CV_error_lam2))]
    print('regul_lam2', regul_lam2)


else:

    while lamda1 > condition1:
        count1 += 1
        print(count1)
        iter = 0
        for g in range(0, m, 5):
            iter += 1
            error_g = CV_Error_KFold(x, y, lamda1, lamda2, g, num_fold, loss)
            CV_error_g = np.append(CV_error_g, error_g)
            g_list = np.append(g_list, g)
        hyp_g = g_list[list(CV_error_g).index(min(CV_error_g))]
        print('hyp_g', hyp_g)
        lamda1 = lamda1 * 0.1
        error_lam1 = CV_Error_KFold(x, y, lamda1, lamda2, hyp_g, num_fold, loss)
        CV_error_lam1 = np.append(CV_error_lam1, error_lam1)
        lamda_list1 = np.append(lamda_list1, lamda1)

    regul_lam1 = lamda_list1[list(CV_error_lam1).index(min(CV_error_lam1))]
    print('regul_lam1', regul_lam1)


if loss== 'denoise':
    plt.plot(lamda_list2, CV_error_lam2, label="lamda")
    plt.xticks(lamda_list2)

else:
    plt.plot(lamda_list1, CV_error_lam1, label="lamda")
    plt.xticks(lamda_list1)

plt.legend()
plt.xscale('log')
plt.gca().invert_xaxis()
plt.show()


'''Plotting the original data, True f and f_hat for least square error, Huber, Denoise and non_decreasing
You need to change the parameters above '''

k = kernel(x, x, hyp_g)

y = np.asarray(y).reshape(-1)
alpha = cp.Variable(k.shape[0])

if loss== 'denoise':
    obj = objective_fn(k, y, alpha, regul_lam1, regul_lam2,  loss)
    cp.Problem(cp.Minimize(obj)).solve()

if loss == 'non_decreasing':
    obj = objective_fn(k, y, alpha, regul_lam1, lamda2,  loss)
    constraint = [((cp.matmul(Diff(k.shape[0]), k)) * alpha) >= 0]
    cp.Problem(cp.Minimize(obj), constraint).solve()

else:
    obj = objective_fn(k, y, alpha, regul_lam1, lamda2,  loss)
    cp.Problem(cp.Minimize(obj)).solve()

alpha = alpha.value

f_hat = predict(k, alpha)


figname = 'HW4_2c.png'
plt.figure(dpi=75)
plt.scatter(x, f, label='True f')
plt.scatter(x, y, label='Original Data')
plt.plot(x, f_hat, color="g", label="f(x)_hat")
plt.title('denoising')
plt.legend()
plt.savefig(figname)
plt.show()
