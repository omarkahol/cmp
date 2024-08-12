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
samples_q = np.genfromtxt("par_samples.csv",delimiter=' ')
samples_h = np.exp(np.genfromtxt("hpar_samples.csv",delimiter=' '))
obs = np.genfromtxt("data.csv",delimiter=' ')
pred = np.genfromtxt("pred.csv",delimiter=' ')

# plot the posterior
g = sns.JointGrid(x=samples_q[:,0],y=samples_q[:,1], space=0, marginal_ticks=False)
g.set_axis_labels('$\\theta_1$', '$\\theta_2$')
g.plot_joint(sns.scatterplot,marker='.',color='black')
ax = g.plot_marginals(sns.kdeplot,fill=True, common_norm=False,
   alpha=.5, linewidth=0.5,color='black')
plt.savefig(os.path.join(os.getcwd(),'posterior.pdf'),bbox_inches='tight')