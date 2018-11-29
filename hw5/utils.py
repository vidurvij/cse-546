import torch as tt
def onehot(x,label):
    xhot = tt.zeros(x.shape[0],len(label))
    for i in range(xhot.shape[0]):
        xhot[i][x[i]] = 1
    return xhot
