# Import seaborn
import seaborn as sns
import torch
import numpy as np
import pandas as pd 
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from numpy import genfromtxt
sns.set_theme()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
"""
df = pd.read_csv("32env250level_relu1L/reward.csv", delimiter=',')
df2 = pd.read_csv("32env250level_tanh1L/reward.csv", delimiter=',')

df["Activation"]  = "ReLU"
df["Smoothed Reward"]  = smooth(df['Average_Reward'],15)
df.drop(df.tail(15).index,inplace=True)
df2["Activation"]  = "Tanh"
df2["Smoothed Reward"]  = smooth(df2['Average_Reward'],15)
df2.drop(df2.tail(15).index,inplace=True)
total_df = df.append(df2)

"""
df = pd.read_csv("32envlev50_tanh/reward.csv", delimiter=',')
df2 = pd.read_csv("32envlev5000_tanh/reward.csv", delimiter=',')
df3 = pd.read_csv("32envlev500_tanh/reward.csv", delimiter=',')
df4 = pd.read_csv("32envlev50000_tanh/reward.csv", delimiter=',')

df["Number of levels"]  = 50
df["Smoothed Reward"]  = smooth(df['Average_Reward'],15)
df.drop(df.tail(15).index,inplace=True)
df2["Number of levels"]  = 5000
df2["Smoothed Reward"]  = smooth(df2['Average_Reward'],15)
df2.drop(df2.tail(15).index,inplace=True)
df3["Number of levels"]  = 500
df3["Smoothed Reward"]  = smooth(df3['Average_Reward'],15)
df3.drop(df3.tail(15).index,inplace=True)
df4["Number of levels"]  = 50000
df4["Smoothed Reward"]  = smooth(df4['Average_Reward'],15)
df4.drop(df4.tail(15).index,inplace=True)

total_df = df.append(df3)
total_df = total_df.append(df2)
total_df = total_df.append(df4)



print(total_df.info())
sns.lineplot(
    data=total_df, x="Step", y="Smoothed Reward", hue="Number of levels", palette="Set2"
)

#plt.plot(df['Step'], df['Average_Reward'])
#plt.plot(df['Step'][0:-5], smooth(df['Average_Reward'],5)[0:-5])
#plt.plot(df2['Step'][0:-5], smooth(df2['Average_Reward'],5)[0:-5])
#plt.plot(df3['Step'][0:-5], smooth(df3['Average_Reward'],5)[0:-5])
#plt.plot(df3['Step'], df3['Average_Reward'])

plt.title("Mean reward and activation function")
#plt.ylabel("Mean Reward")
#plt.xlabel("Training step (*10e3)")

plt.show()
plt.savefig("output_tanh_vs_relu.png")
plt.close()

#np.mean( np.array([ old_set, new_set ]), axis=0 )