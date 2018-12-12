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
print(alpha.shape)
lam_vals1 = np.array([0, .1])
# lam_vals2 = np.logspace(1,10,10)
# lam_vals2 = np.array([1,1])
# lam_vals = np.array(  [1,2])
lam = cvx.Parameter(nonneg = True)
# lam2 = cvx.Parameter(nonneg = True)
Plot = Plotter(xr,yr)


### Problem Setup

# objective = cvx.Problem(cvx.Minimize(cvx.sum(y-K@alpha)))


lam_vals = np.geomspace(.0000001,100,10)
gammas = np.arange(500,2500,100)
gammas = [100]
lam_vals = [lam_vals[4]]
print(gammas, lam_vals)
# Plot.__init__
## One-Out Cross
# for k in range(lam_vals2.shape[0]):
#     lam2.value = lam_vals2[k]
losses = []
for k in tqdm(range(len(gammas))):
    gamma = gammas[k]
    for j in tqdm(range(len(lam_vals))):
        lam.value = lam_vals[j]
        loss = 0
        for i in (range(xr.shape[0])):
            x = np.delete(xr,i).reshape(xr.shape[0]-1,1)
            y = np.delete(yr,i).reshape(xr.shape[0]-1,1)
            K = Km(x,gamma)
            D = D_matrix(x.shape[0])

            square = cvx.square(cvx.norm(K@alpha-y,1))
            # D * K
            constraint1 = cvx.quad_form(alpha,K)
            constraint2 = [D@K*alpha >= 0]            # l =  +
            objective = cvx.Problem(cvx.Minimize(square + lam * constraint1),constraint2)
            objective.solve()
            # print(alpha.value)
            loss += Plot.loss(alpha.value,x,gamma,i)
        loss /= xr.shape[0]
        # losses[k][j] = loss
        Plot.overlay(alpha.value,x,lam.value,loss,gamma)
print(losses)
plt.legend()
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Question 4 with $\gamma$ = "+ str(gamma))
# plt.show()
plt.savefig("Question2d.png")
