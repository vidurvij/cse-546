import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


n = 500
d = 1000
K = 100
X = np.random.randn(n, d)
W = (np.arange(1, K+1, 1)/100)
W = np.append(W, np.zeros((1, d-K))).reshape(d, 1)
# print(W[0:K])

mu, sigma = 0, 1
epsilon = np.random.normal(mu, sigma, (n, 1))

Y = (X @ W) + epsilon
# print(Y - np.mean(Y))

# print(X.shape, W.shape, Y.shape)

Y_demean = (Y - np.mean(Y))

# print(Y_demean.shape)
lamda_max = np.amax(2*abs((np.dot(X.T, Y_demean).T)))


def soft_threshold(c, a, lamda):
    '''Soft threshold function used for normalized data and lasso regression'''
    if c < - lamda:
        return (c + lamda)/a
    elif c > lamda:
        return (c - lamda)/a
    else:
        return 0



# R = np.zeros((n, 1))

def coordinate_Descent_Lasso(lamda):

    count = 0
    W = np.zeros((d, 1))
    W_old = np.ones((d, 1))

    objective_value_initial =10000000
    # conv_condition = 10
    while np.max(np.abs(W - W_old)) > 0.1:

        W_old = np.copy(W)
        b = np.mean(Y - X @ W)
        # print('b_old', b)
        Value_objective_old = np.dot(((X @ W) + b - Y).T, ((X @ W) + b - Y)) + (lamda * np.sum(W))
        # print('Value_objective_old', Value_objective_old)
        # print('-----b:', b)
        W = W.reshape(W.shape[0], 1)
        # print("------",W.shape,lamda.shape)

        for k in (np.arange(d)):
                W_old_exclud = np.copy(W)
                W_old_exclud[k] = 0
                a = 2*np.sum(X[:, k]**2)
                # print('-----a:', a)
                R = Y - (b + X @ W_old_exclud)
                c = 2*(X.T @ R)[k]
                # print('-----c:',c)
                # print("C:---",c.shape)
                # if c < -lamda or c>lamda:
                #     print('*********')
                W_new = soft_threshold(c, a, lamda)
                W[k] = W_new
                # print('W[k]', W[k])
                # value_obj = (((X @ W) + b - Y)[k])**2
                # print('element c:', c, 'element a:', a, 'W_new:', W_hat, 'size W_hat:', W_hat.shape)
        b = np.mean(Y - X @ W)
        # print('b_new', b)

        # value_objective = np.dot(((X @ W) + b - Y).T, (X @ W) + b - Y) + lamda * np.sum(np.abs(W))
        # print('value_objective', value_objective)
        # print('diff_value:',value_objective - Value_objective_old)
        # if value_objective > Value_objective_old:
        #     print('Error')
        # conv_condition = np.max(W - W_old)
        count = count+1
        value_objective = np.dot(((X @ W) + b - Y).T, (X @ W) + b - Y) + lamda * np.sum(np.abs(W))
        if value_objective > objective_value_initial:
            print
        # print(value_objective)
        # lamda= lamda/1.5
    return W, value_objective, count

# costs_u = []
ws = []
costs = []
counts = []
lamdas = []
fdrs = []
tdrs = []
lamda = lamda_max
for i in tqdm(range(200)):
    w , cost, count = coordinate_Descent_Lasso(lamda)
    FDR = np.count_nonzero(w[K + 1:d])
    TDR = np.count_nonzero(w[0:K])/np.count_nonzero(W)
    num_nonZero_W = np.count_nonzero(np.count_nonzero(W))
    #print(TDR/(num_nonZero_W+0.00001), FDR/(num_nonZero_W+0.00001))
    costs.append(np.asscalar(cost))
    #ws.append(w)
    fdrs.append(FDR)
    tdrs.append(TDR)
    counts.append(count)
    lamdas.append(lamda)
    lamda = lamda/1.5
#plt.plot(costs)
plt.plot(fdrs,tdrs)
plt.show()

exit()








print(coordinate_Descent_Lasso(lamda_max))
exit()
u, value_objective_u, count_u= coordinate_Descent_Lasso(lamda_max)
v, value_objective_v, count_v= coordinate_Descent_Lasso(lamda_max/1.5)
#
# def lamda_list():
#     for i in
#     lamda[i] = lamda_max/(1.5)^i

FDR = np.count_nonzero(coordinate_Descent_Lasso(lamda)[0][K+1:d])
TDR = np.count_nonzero(coordinate_Descent_Lasso(lamda)[0][0:K])


exit()
lamda_list = [value_objective_u,value_objective_v]
for i in range(len(lamda_list)-1):
    #u, value_objective_u = coordinate_Descent_Lasso(lamda)
    if lamda_list[i][len(lamda_list[i])] > lamda_list[i+1][0]:
        print('Error')

exit()


print('result', value_objective_u, count_u, value_objective_v, count_v)


# costs_u.append(np.asscalar(cost_u))

# print('result', u, value_objective_u, count_u)

print('result', value_objective_u, count_u, value_objective_v, count_v)



exit()
costs_u = []
costs_v = []

u, cost_u= coordinate_Descent_Lasso(lamda_max)
v, cost_v= coordinate_Descent_Lasso(lamda_max/1.5)
print(cost_u.shape)

costs_u.append(np.asscalar(cost_u))
costs_v.append(np.asscalar(cost_u))
print(costs_u)
# exit()

print('costs_u', costs_u)
plt.plot(costs_u)
plt.show()
print(np.count_nonzero(u))
# num_nonZero_W = np.count_nonzero(coordinate_Descent_Lasso(lamda))

exit()

costs= []



# def decreasing_constant_ratio(lamda):
#     while num_nonZero_W <= d:
#
#
#     lamda = np.copy(lamda_max/1.5)
#     W = coordinate_Descent_Lasso(lamda)
#     print(W.shape)
#     num_nonZero_W = np.count_nonzero(W)
#     print('num_nonZero_W', num_nonZero_W)
#


lamda_list = list()

# lamda = np.logspace(0,4,300)/10 #Range of lambda values

# for l in lamda:
#     W = coordinate_Descent_Lasso(lamda)
#     lamda_list.append(lamda)

# while num_nonZero_W <= d:
#     lamda = np.copy(lamda_max/1.5)
#     W = coordinate_Descent_Lasso(lamda)
#     print(W.shape)
#     num_nonZero_W = np.count_nonzero(W)
#     print('num_nonZero_W', num_nonZero_W)
#
