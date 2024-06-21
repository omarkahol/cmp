import math as mt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

font = {"text.usetex": True,
        'font.weight' : 'normal',
        'font.size'   : 20,
        "font.family": "serif"}
plt.rcParams.update(font)


# retrieve data
data = np.genfromtxt('pred.csv', delimiter=' ')
x_pred = data[:,0]
y_pred_mean = data[:,1]
y_pred_95 = 2*np.sqrt(abs(data[:,2]))

# plot the prediction
fig, ax = plt.subplots(figsize=(12, 8))
p2, = plt.plot(x_pred,y_pred_mean,'r-')
plt.fill_between(x_pred,y_pred_mean-y_pred_95, y_pred_mean+y_pred_95, alpha=0.4, color="red",lw=0)
plt.savefig(os.path.join(os.getcwd(),'pred_ls.pdf'))
