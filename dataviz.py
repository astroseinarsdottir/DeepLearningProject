# Import seaborn
import seaborn as sns
import torch
import numpy as np
import pandas as pd 
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from numpy import genfromtxt
sns.set_theme()




my_data = genfromtxt("50_level_8e6/mean_reward.csv", delimiter=',')
my_data2 = genfromtxt("250_level_8e6/mean_reward.csv", delimiter=',')
my_data3= genfromtxt("500_level_8e6/mean_reward.csv", delimiter=',')

x = np.linspace(0, len(my_data), len(my_data))
print(len(x), len(my_data))
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

merged = [smooth(my_data,20)[0:-10],smooth(my_data2,20)[0:-10],smooth(my_data3,20)[0:-10]]

sns_plot = sns.lineplot(data=smooth(my_data,20)[0:-10], label="N = 50", color='mediumvioletred')
sns_plot = sns.lineplot(data=smooth(my_data2,20)[0:-10], label="N = 250")
sns_plot = sns.lineplot(data=smooth(my_data3,20)[0:-10],label="N = 500")

plt.title("Mean reward variation with number of levels")
plt.ylabel("Mean Reward")
plt.xlabel("Training step (*10e3)")

fig = sns_plot.get_figure()
fig.savefig("output.png")

#np.mean( np.array([ old_set, new_set ]), axis=0 )