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
lam_vals = np.array([.0, .2])
# lam_vals = np.array(  [1,2])
lam = cvx.Parameter(nonneg=True)
Plot = Plotter(xr,yr)

lam_vals = np.geomspace(.0000001,100,10)
gammas = np.arange(500,2500,100)
gammas = [100]
lam_vals = [lam_vals[4]]
print(gammas, lam_vals)
### Problem Setup

# objective = cvx.Problem(cvx.Minimize(cvx.sum(y-K@alpha)))

# Plot.__init__
## One-Out Cross
losses = []
# Plot.__init__
## One-Out Cross
# losses = np.zeros((gammas.shape[0],lam_vals.shape[0]))
for k in tqdm(range(len(gammas))):
    gamma = gammas[k]
    for j in tqdm(range(len(lam_vals))):
        lam.value = lam_vals[j]
        loss = 0
        for i in (range(xr.shape[0])):
            x = np.delete(xr,i).reshape(xr.shape[0]-1,1)
            y = np.delete(yr,i).reshape(xr.shape[0]-1,1)
            K = Km(x,gamma)
            # print(yr.shape)
            square = cvx.sum(cvx.huber(y-K@alpha))
            constraint = cvx.quad_form(alpha,K)
            # l =  +
            objective = cvx.Problem(cvx.Minimize(square + lam * constraint))
            objective.solve()
            # print(alpha.value)
            loss += Plot.loss(alpha.value,x,gamma,i)
        loss /= xr.shape[0]
        # losses[k][j] = loss
        Plot.overlay(alpha.value,x,lam.value,loss,gamma)
# print(losses)
# print(np.argmin(losses))
plt.legend()
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Kernel with HUber Loss with $\gamma$ = "+str(gamma))
plt.savefig("Question2b.png")
# plt.show()

#
# square = cvx.sum(cvx.huber(y-K@alpha))
# # constraint = cvx.quad_form(alpha,K)
# # # l =  +
# objective = cvx.Problem(cvx.Minimize(square + lam * constraint))
