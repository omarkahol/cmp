from fig_settings import *
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update(tex_fonts)


# retrieve data
data = np.genfromtxt('pred.csv', delimiter=' ')

# plot the prediction
fig, ax = plt.subplots(1,1, figsize=set_size('thesis'))
ax.plot(data[:,0], data[:,1], 'k-', label='True function')
ax.plot(data[:,0], data[:,3], 'r--', label='Prediction')
ax.fill_between(data[:,0], data[:,3]-data[:,5], data[:,3]+data[:,5], color='r', alpha=0.2, lw=0)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
plt.show()


fig, ax = plt.subplots(1,1, figsize=set_size('thesis'))
ax.plot(data[:,0], data[:,2], 'k-', label='True function')
ax.plot(data[:,0], data[:,4], 'r--', label='Prediction')
ax.fill_between(data[:,0], data[:,4]-data[:,6], data[:,4]+data[:,6], color='r', alpha=0.2, lw=0)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
plt.show()