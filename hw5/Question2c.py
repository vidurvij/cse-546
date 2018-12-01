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
lam_vals1 = np.logspace(1,10,10)
lam_vals2 = np.logspace(1,10,10)
lam_vals2 = np.array([1,1])
# lam_vals = np.array(  [1,2])
lam1 = cvx.Parameter(nonneg = True)
lam2 = cvx.Parameter(nonneg = True)
Plot = Plotter(xr,yr)


### Problem Setup

# objective = cvx.Problem(cvx.Minimize(cvx.sum(y-K@alpha)))

# Plot.__init__
## One-Out Cross
for k in range(lam_vals2.shape[0]):
    lam2.value = lam_vals2[k]
    for i in tqdm(range(xr.shape[0])):
        x = np.delete(xr,i).reshape(xr.shape[0]-1,1)
        y = np.delete(yr,i).reshape(xr.shape[0]-1,1)
        D = D_matrix(x.shape[0])
        for j in tqdm(range(lam_vals1.shape[0])):
            lam1.value = lam_vals1[j]
            K = Km(x)

            # print(yr.shapz
            # exit()
            square = cvx.square(cvx.norm(K@alpha-y,1))
            # D * K
            constraint1 = cvx.quad_form(alpha,K)
            constraint2 = cvx.norm(D@K*alpha,1)
            # l =  +
            objective = cvx.Problem(cvx.Minimize(square + lam1 * constraint1 + lam2 * constraint2))
            objective.solve()
            Plot.overlay(alpha.value,x)
plt.show()
