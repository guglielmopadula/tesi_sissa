import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

names=["AE","VAE","AAE","BEGAN"]


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 10
})

var_tot=np.zeros(5)
mmd_tensor_tot=np.zeros(4)
mmd_area_tot=np.zeros(4)
rec_error_tot=np.zeros(4)
kid_tot=np.zeros(4)
mmd_drag_tot=np.zeros(4)
mmd_momz_tot=np.zeros(4)

fig2,ax2=plt.subplots(2,2)
#ax2.set_title("Moment over z")
fig2.suptitle("Moment over z")

for i in range(len(names)):
    name=names[i]
    momz_data=np.load("physical_quantities/momz_data.npy").reshape(-1)
    momz_sampled=np.load("physical_quantities/momz_"+name+".npy").reshape(-1)
    _=ax2[i//2,i%2].set_title(name)
    _=ax2[i//2,i%2].hist(momz_data,8,label='real',histtype='step',linestyle='solid',density=True)
    _=ax2[i//2,i%2].hist(momz_sampled,8,label=name,histtype='step',linestyle='dotted',density=True)
    ax2[i//2,i%2].grid(True,which='both')

fig2.tight_layout()
fig2.savefig("./plots/multiplot.png",dpi=600)

