import numpy as np
from Kernel import Kernel
import matplotlib.pyplot as plt
np.random.seed(9)

def data():
    xs = []
    ys = []
    for i in rawnge(300):
        x = np.random.uniform()
        y = 4 * np.sin(np.pi*x) * np.cos(6*np.pi*(x**2)) + np.random.normal()
        xs.append(x)
        ys.append(y)

    return np.array(xs) , np.array(ys)
x, y = data()
A = Kernel(x.reshape(x.shape[0],1),y,2,25)
# A.cross_validation(.3)
alpha = A.Regression(A.train,A.label)
y = np.zeros(A.label.shape)
for i in range(A.train.shape[0]):
    y[i] = A.predict(A.train[i],A.train,alpha)
plt.scatter(A.train,y, label = "Predicted")
plt.scatter(A.train, A.label , label = "Actual")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
plt.savefig ("Kernel/kernel1-300.png")
# A.cross_validation(.0000001)
# A.Bootstrap(300)
