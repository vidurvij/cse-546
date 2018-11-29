import numpy as np
import matplotlib.pyplot as plt

a = np.load("ONE.npy")
# b = np.sqrt(np.load("Test.npy")-1)
# x = [1,2,5,10,20,50]
print(a)
plt.plot(a)
# plt.plot(x,np.flip(b))
# plt.gca().invert_xaxis()
plt.grid()
plt.xlabel("Number of Features")
plt.ylabel("Error")
plt.title("Features vs L1")
plt.show()
# plt.savefig("L1.png")
