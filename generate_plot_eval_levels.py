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


df = pd.read_csv("plot_models_eval.csv", delimiter=',')

    
fig_dims = (7, 5)
fig, ax = plt.subplots(figsize=fig_dims)
#g= sns.lineplot(
#    data=df, x="Id model", y="Average_Reward", marker="o"
#)
g= sns.barplot(
    data=df, x="Average_Reward", y="Run name",capsize=.2,palette="Set2"
)

box = g.get_position()
g.set_position([box.x0*2, box.y0*1.5, box.width * 0.75, box.height]) # resize position
g.tick_params(labelsize=20)

#plt.ylabel("Mean Reward")
#plt.xlabel("Training step (*10e3)")

#g.set_xscale('log')
#g.set_xticks(["BL","BL tanh", "BL ReluReg"])
#g.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel("Mean Reward")
plt.ylabel("")
ax.set_yticklabels(["Baseline (BL)","BL + Tanh","BL + ReLUReg","Dv + Tanh","Dv + ReLUReg"],fontsize=15)
plt.show()
plt.savefig("compare_eval.png")
plt.close()

"""
df = pd.read_csv("32envlev50_tanh/reward.csv", delimiter=',')
df2 = pd.read_csv("32envlev5000_tanh/reward.csv", delimiter=',')
df3 = pd.read_csv("32envlev500_tanh/reward.csv", delimiter=',')
df4 = pd.read_csv("32envlev50000_tanh/reward.csv", delimiter=',')


df["Number of levels"]  = 50
df["Reward"]  = smooth(df['Average_Reward'],15)
df.drop(df.tail(15).index,inplace=True)
df2["Number of levels"]  = 5000
df2["Reward"]  = smooth(df2['Average_Reward'],15)
df2.drop(df2.tail(15).index,inplace=True)
df3["Number of levels"]  = 500
df3["Reward"]  = smooth(df3['Average_Reward'],15)
df3.drop(df3.tail(15).index,inplace=True)
df4["Number of levels"]  = 50000
df4["Reward"]  = smooth(df4['Average_Reward'],15)
df4.drop(df4.tail(15).index,inplace=True)

total_df = df.append(df3)
total_df = total_df.append(df2)
total_df = total_df.append(df4)

lognorm = LogNorm(vmin=1, vmax=50000)

print(total_df.info())
fig_dims = (10, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.lineplot(
    data=total_df, x="Step", y="Reward", hue="Number of levels", palette="Blues",hue_norm=lognorm
)

#plt.plot(df['Step'], df['Average_Reward'])
#plt.plot(df['Step'][0:-5], smooth(df['Average_Reward'],5)[0:-5])
#plt.plot(df2['Step'][0:-5], smooth(df2['Average_Reward'],5)[0:-5])
#plt.plot(df3['Step'][0:-5], smooth(df3['Average_Reward'],5)[0:-5])
#plt.plot(df3['Step'], df3['Average_Reward'])

plt.title("Influence of level number on training with coinrun")
#plt.ylabel("Mean Reward")
#plt.xlabel("Training step (*10e3)")

plt.show()
plt.savefig("output_tanh_vs_relu.png")
plt.close()
"""

#np.mean( np.array([ old_set, new_set ]), axis=0 )