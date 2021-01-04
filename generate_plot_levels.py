# Import seaborn
import seaborn as sns
import torch
import numpy as np
import pandas as pd 
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from numpy import genfromtxt
sns.set_theme()
from matplotlib.colors import LogNorm
import os


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Name of the legend title
legend_name = "Episode size"

# First element is the name of the folder, second element is the legend you want to give it
names = [
    #['50000_levels_hard_dv','50000'],
    #['50000_levels_hard_dvdpRELU','50000 4'],
    ['50000_model_deep_value_autoeval_91','256'],
    ['50000_model_deep_value_autoeval_512_91','512'],
    ['50000_model_deep_value_autoeval_128_91','128']
]

markers= ["^","*","s","p","d"]

df_total = pd.DataFrame()

for name in names:
    #The more you smooth, the more it eats the tail of the data
    smooth_coeff = 10
    df = pd.read_csv(name[0]+"/validations_model.csv", delimiter=',')
    df[legend_name]  = name[1]
    df["Reward"]  = smooth(df['Average_Reward'],smooth_coeff)
    df.drop(df.tail(smooth_coeff).index,inplace=True)
    if df_total.empty:
        df_total = df
    else:
        df_total = df_total.append(df)
    
# EDIT: I Needed to ad the fig
fig, ax1 = plt.subplots(1,1)
g = sns.lineplot(
    data=df_total, x="Step", y="Reward", hue=legend_name, style=legend_name, linewidth=3, palette="Set2", markers=True, dashes=False, markevery=[-1],markersize=15
)
# EDIT: 
# Removed 'ax' from T.W.'s answer here aswell:
box = g.get_position()
g.set_position([box.x0*1.05, box.y0*1.5, box.width * 0.85, box.height]) # resize position

# Put a legend to the right side
g.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), ncol=1)

g.tick_params(labelsize=20)
# Turn off tick labels


#plt.title("Training on coinrun")
g.set_ylabel("Reward on evaluation",fontsize=20)
g.set_xlabel("Steps",fontsize=20)
plt.xticks([0,5e6, 10e6], ('0','5M', '10M'))
# Plot using seaborn

plt.show()
plt.savefig("compare_ep_step.png")
plt.close()