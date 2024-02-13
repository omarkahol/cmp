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

# plot the prediction
fig, ax = plt.subplots(figsize=(12, 8))
p0, =plt.plot(obs[:,0],obs[:,1],'k+',label='Experimental points')
p1, = plt.plot(pred[:,0],pred[:,1],'b-',label = 'calibrated model')
p2, = plt.plot(pred[:,0],pred[:,4],'r-',label = 'corrected model')
plt.fill_between(pred[:,0],pred[:,2], pred[:,3], alpha=0.1, color="blue",lw=0)
plt.legend(handles=[p0,p1,p2],loc='lower right')
plt.savefig(os.path.join(os.getcwd(),'pred_ls.pdf'))

# plot the posterior
g = sns.JointGrid(x=samples_q[:,0],y=samples_q[:,1], space=0, marginal_ticks=False)
g.set_axis_labels('$\\theta_1$', '$\\theta_2$')
g.plot_joint(sns.scatterplot,marker='.',color='black')
ax = g.plot_marginals(sns.kdeplot,fill=True, common_norm=False,
   alpha=.5, linewidth=0.5,color='black')
plt.savefig(os.path.join(os.getcwd(),'posterior.pdf'),bbox_inches='tight')

# plot the hyperparameter samples
fig, ax = plt.subplots(3, 1,figsize=(8,16))
ax[0].set_xlabel('$\\sigma_e$')
ax[1].set_xlabel('$\\sigma$')
ax[2].set_xlabel('$l$')

g1=sns.histplot(samples_h[:,0],kde=False,ax=ax[0],lw=0,stat='percent')
g2=sns.histplot(samples_h[:,1],kde=False,ax=ax[1],lw=0,stat='percent')
g3=sns.histplot(samples_h[:,2],kde=False,ax=ax[2],lw=0,stat='percent')

plt.savefig(os.path.join(os.getcwd(),'samples_h.pdf'),bbox_inches='tight')