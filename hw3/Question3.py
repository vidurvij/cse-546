import numpy as np
import scipy as sp

n = 500
d = 1000
k = 100
def random_data(): # Generating Random Data with n features and d examples
    x = np.random.randn(500,1000)
    w = np.zeros((500,1))
    for i in range(k):
        w[i] = (i+1)/k
    print (w)
    y = w.T@x + np.random.randn(500,1000)
    return x,y

random_data()
