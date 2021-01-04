# Import seaborn
import seaborn as sns
import torch
import numpy as np
import pandas as pd 
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib
from numpy import genfromtxt
sns.set_theme()
from matplotlib.colors import LogNorm
import os


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Name of the legend title
legend_name = "N. Levels"

# First element is the name of the folder, second element is the legend you want to give it


df = pd.read_csv("plot_n_level.csv", delimiter=',')

    
fig_dims = (7, 5)
fig, ax = plt.subplots(figsize=fig_dims)
#g= sns.lineplot(
#    data=df, x="Id model", y="Average_Reward", marker="o"
#)
g= sns.barplot(
    data=df, x="Average_Reward", y="Run name",capsize=.2,palette="crest"
)

box = g.get_position()
g.set_position([box.x0*2.5, box.y0*1.5, box.width * 0.8, box.height*0.7]) # resize position
g.tick_params(labelsize=20)

#plt.ylabel("Mean Reward")
#plt.xlabel("Training step (*10e3)")

#g.set_xscale('log')
#g.set_xticks(["BL","BL tanh", "BL ReluReg"])
#g.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel("Mean Reward")
plt.ylabel("")
ax.set_yticklabels(["50 Levels","500 Levels","5000 Levels","50000 Levels"],fontsize=15)

plt.show()
plt.savefig("compare_eval_noga.png")
plt.close()