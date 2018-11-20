import numpy as np

def data():
    data = []
    for i in range(30):
        x = np.random.uniform()
        y = 4 * np.sin(np.pi*x) * np.cos(6*np.pi*(x**2)) + np.random.normal()
        data.append((x,y))

    return data
