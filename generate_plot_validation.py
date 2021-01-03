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
legend_name = "N. levels"

# First element is the name of the folder, second element is the legend you want to give it

name = '500000_model_deep_value_autoeval_91'
markers= ["^","*","s","p","d"]

df_total = pd.DataFrame()

    #The more you smooth, the more it eats the tail of the data
smooth_coeff = 10

df = pd.read_csv(name+"/validations_model.csv", delimiter=',')
#df[legend_name]  = name[1]
df["Reward"]  = smooth(df['Average_Reward'],smooth_coeff)
df.drop(df.tail(smooth_coeff).index,inplace=True)
if df_total.empty:
    df_total = df
else:
    df_total = df_total.append(df)
    
# EDIT: I Needed to ad the fig
fig, ax1 = plt.subplots(1,1)
g = sns.lineplot(
    data=df_total, x="Step", y="Reward", linewidth=3, palette="crest", markers=True, dashes=False, markevery=[-1],markersize=15
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
g.set_ylabel("Reward on validation",fontsize=20)
g.set_xlabel("Steps",fontsize=20)
plt.xticks([0,5e6, 10e6], ('0','5M', '10M'))
# Plot using seaborn

plt.show()
plt.savefig("validationscore.png")
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