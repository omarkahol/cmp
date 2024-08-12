import numpy as np
import matplotlib.pyplot as plt


# Load data
data = np.loadtxt('chain.txt')


# Plot data
plt.plot(data, 'k-', lw=0.5)
plt.xlabel('Iteration')
plt.ylabel('Parameter')
plt.show()


# Plot histogram
plt.hist(data, color='k',bins=int(np.sqrt(len(data))))
plt.xlim([-1.5,1.5])
plt.xlabel('Parameter')
plt.ylabel('Frequency')
plt.show()
