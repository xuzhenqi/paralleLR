import matplotlib.pyplot as plt
import sys
import numpy as np

if __name__ == '__main__':
    filename = sys.argv[1]
    data = np.loadtxt(filename)
    print data.shape, len(data)
    plt.plot(data[:, 0], data[:, 1], '.')
    plt.show()
