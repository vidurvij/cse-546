import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import seaborn as sns
import pandas as pa
import itertools
def Generate():
    mu = [np.array([1,2]),np.array([-1,1]),np.array([2,-2])]
    sigma = [np.array([[1,0],[0,2]]),np.array([[2,-1.8],[-1.8,2]]),np.array([[3,1],[1,2]])]
    plots = []
    num = []
    for i,(m,s) in enumerate(zip(mu,sigma)):
        a = np.random.randn(2,100)
        points = s@a+np.reshape(m,(2,1))
        #points=points+np.reshape(m,(2,1))
        #plot = sns.relplot(x = "x", y = "y",data = data, marker = '^', style = 'status')
        #plots.append(plot)
        num.append(points)

    return num

def stats(num):
    stats = []
    for m in num:
        mean = np.mean(m,axis = 1)
        covar = np.cov(m)
        stats.append((mean,covar))
    return stats

def eigen(stats):
    eigen = []
    eigenv = []
    for i,(mean,covar) in enumerate(stats):
        eig,eigv = np.linalg.eig(covar)
        eigen.append(eig)
        eigenv.append(eigv)
    return eigen,eigenv

def scatter(num,stats,eigen,eigenv):
    plots = []
    for i,(X,(m,s),eig,eigv) in enumerate(zip(num,stats,eigen,eigenv)):
        X_new = np.divide((eigv.T@(X-np.reshape(m,(2,1)))),np.reshape(1/np.sqrt(eig),(2,1)))
        data1 = pa.DataFrame(X_new.T, columns = ['x','y'])
        data1['status'] = "Modified"
        data2 = pa.DataFrame(X.T, columns = ['x','y'])
        data2['status'] = "Original"
        data = pa.concat([data1,data2])
        plot = sns.relplot(x = "x", y = "y",data = data,  hue = 'status',style = 'status', markers = {"Original":'^',"Modified":'o'})
        plots.append(plot)
    return plots

def plot(plots,eigen,eigenv):
    for i,((mean,covar),plot,eig,eigv) in enumerate(zip(stats,plots,eigen,eigenv)):
        l1 = lines.Line2D([mean[0],mean[0]+np.sqrt(eig[0])*eigv[0][0]],[mean[1],mean[1]+np.sqrt(eig[0])*eigv[1][0]],color = 'g')
        l2 = lines.Line2D([mean[0],mean[0]+np.sqrt(eig[1])*eigv[0][1]],[mean[1],mean[1]+np.sqrt(eig[1])*eigv[1][1]],color='p')
        plot.ax.add_line(l1)
        plot.ax.add_line(l2)
        plot.savefig("Question1-"+str(i)+".png")

if __name__ == "__main__":
    num = Generate()
    stats = stats(num)
    eigen,eigenv = eigen(stats)
    plots = rescatter(num,stats,eigen,eigenv)
    plot(plots,eigen,eigenv)
