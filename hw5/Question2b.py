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
lam_vals = np.logspace(1,10,10)
# lam_vals = np.array(  [1,2])
lam = cvx.Parameter(nonneg=True)
Plot = Plotter(xr,yr)


### Problem Setup

# objective = cvx.Problem(cvx.Minimize(cvx.sum(y-K@alpha)))

# Plot.__init__
## One-Out Cross
for i in tqdm(range(xr.shape[0])):
    x = np.delete(xr,i).reshape(xr.shape[0]-1,1)
    y = np.delete(yr,i).reshape(xr.shape[0]-1,1)
    for j in tqdm(range(lam_vals.shape[0])):
        lam.value = lam_vals[j]
        K = Km(x)

        # print(yr.shape)
        # exit()
        square = cvx.sum(cvx.huber(y-K@alpha))
        constraint = cvx.quad_form(alpha,K)
        # l =  +
        objective = cvx.Problem(cvx.Minimize(square + lam * constraint))
        objective.solve()
        Plot.overlay(alpha.value,x)
plt.show()
