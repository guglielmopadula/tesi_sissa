import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from PIL import Image

names=["AE","VAE","AAE","BEGAN","DM","EBM","NF"]


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 8
})


fig2,ax2=plt.subplots(3,3)


for i in range(6):
    name=names[i]
    energy_data=np.load("physical_quantities/energy_data.npy").reshape(-1)
    energy_sampled=np.load("physical_quantities/energy_"+name+".npy").reshape(-1)
    ax2[i//3,i%3].locator_params(nbins=5)
    _=ax2[i//3,i%3].set_title(name)
    _=ax2[i//3,i%3].hist(energy_data,8,label='real',histtype='step',linestyle='solid',density=True)
    _=ax2[i//3,i%3].hist(energy_sampled,8,label=name,histtype='step',linestyle='dotted',density=True)
    ax2[i//3,i%3].grid(True,which='both')
x_left, x_right = ax2[0,0].get_xlim()
y_low, y_high = ax2[0,0].get_ylim()
    #ax2[i//3,i%3].set_box_aspect(1)

name="NF"
energy_data=np.load("physical_quantities/energy_data.npy").reshape(-1)
energy_sampled=np.load("physical_quantities/energy_"+name+".npy").reshape(-1)
ax2[2,1].locator_params(nbins=5)
_=ax2[2,1].set_title(name)
_=ax2[2,1].hist(energy_data,8,label='real',histtype='step',linestyle='solid',density=True)
_=ax2[2,1].hist(energy_sampled,8,label=name,histtype='step',linestyle='dotted',density=True)
ax2[2,1].grid(True,which='both')
fig2.suptitle("energy")
fig2.delaxes(ax2[2,0])
fig2.delaxes(ax2[2,2])
fig2.tight_layout()
fig2.savefig("./plots/multiplot.png",dpi=600)

