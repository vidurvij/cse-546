import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from utils import *
from tqdm import tqdm
# Data Generation
xr, yr = Data_Generation()


### Constant Setup
# K = np.zeros((x.shape[0],x.shape[0]))
# Kf = Kernel()
# K = Km(xr)
alpha = cvx.Variable((xr.shape[0]-1,1))
# print(alpha.shape)
lam_vals1 = np.array([0])
lam_vals2 = np.array([5.5])
# lam_vals2 = np.array([1,1])
gammas = np.arange(500,2500,100)
lam_vals1 = np.geomspace(.0000001,100,10)
lam_vals2 = np.geomspace(.0000001,10000000,15)
lam_vals1 = [lam_vals1[3]]
gammas = [100]
lam_vals2 = [lam_vals2[3]]
lam1 = cvx.Parameter(nonneg = True)
lam2 = cvx.Parameter(nonneg = True)
Plot = Plotter(xr,yr)
losses = []

### Problem Setup
# objective = cvx.Problem(cvx.Minimize(cvx.sum(y-K@alpha)))

# Plot.__init__
## One-Out Cross
for o in tqdm(range(len(gammas))):
    gamma = gammas[o]
    for k in tqdm(range(len(lam_vals1))):
        lam1.value = lam_vals1[k]
        for j in tqdm(range(len(lam_vals2))):
            lam2.value = lam_vals2[j]
            # print(lam2.value,lam1.value,gamma)
            loss = 0
            for i in tqdm(range(xr.shape[0])):
                x = np.delete(xr,i).reshape(xr.shape[0]-1,1)
                y = np.delete(yr,i).reshape(xr.shape[0]-1,1)
                D = D_matrix(x.shape[0])

                K = Km(x)

                # print(K.shape)
                # exit()
                square = cvx.square(cvx.norm(K@alpha-y,1))
                # D * K
                constraint1 = cvx.quad_form(alpha,K)
                constraint2 = cvx.norm(D@K*alpha,1)
                # l =  +
                objective = cvx.Problem(cvx.Minimize(square + lam1 * constraint1 + lam2 * constraint2))
                objective.solve()
                # print(type(alpha.value))
                loss += Plot.loss(alpha.value,x,gamma,i)
            loss /= xr.shape[0]
            losses.append(loss)
            Plot.overlay2(alpha.value,x,lam1.value,lam2.value,loss,gamma)
print("@@@@@@@@@@@@@@@@@@@@@",np.argmin(np.array(losses)))
plt.legend()
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Question 3 with $\gamma$ = " + str(gamma))
plt.savefig("Question2c.png")
