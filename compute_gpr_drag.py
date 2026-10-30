from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from tqdm import trange
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from time import time
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer

def max_error(y_true, y_pred):
    return np.max(np.abs(y_true.reshape(-1,1) - y_pred.reshape(-1,1)))


gpr=GaussianProcessRegressor()
AE_latent=np.load("tesi_sissa/hull/latent_variables/AE_latent.npy")[:100]
VAE_latent=np.load("tesi_sissa/hull/latent_variables/VAE_latent.npy")[:100]
AAE_latent=np.load("tesi_sissa/hull/latent_variables/AAE_latent.npy")[:100]
BEGAN_latent=np.load("tesi_sissa/hull/latent_variables/BEGAN_latent.npy")[:100]
data_latent=np.load("tesi_sissa/hull/latent_variables/data_latent.npy")[:100][:,:-4]


AE_drag=np.load("tesi_sissa/hull/physical_quantities/drag_AE.npy")
VAE_drag=np.load("tesi_sissa/hull/physical_quantities/drag_VAE.npy")
AAE_drag=np.load("tesi_sissa/hull/physical_quantities/drag_AAE.npy")
BEGAN_drag=np.load("tesi_sissa/hull/physical_quantities/drag_BEGAN.npy")
data_drag=np.load("tesi_sissa/hull/physical_quantities/drag_data.npy")

def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

AE_latent=normalize(AE_latent)
AAE_latent=normalize(AAE_latent)
BEGAN_latent=normalize(BEGAN_latent)
VAE_latent=normalize(VAE_latent)
data_latent=normalize(data_latent)

AE_drag=normalize(AE_drag)
AAE_drag=normalize(AAE_drag)
BEGAN_drag=normalize(BEGAN_drag)
VAE_drag=normalize(VAE_drag)
data_drag=normalize(data_drag)



loo=LeaveOneOut()

err_ae=np.inf
err_vae=np.inf
err_aae=np.inf
err_began=np.inf
err_data=np.inf



all_err=np.zeros(5)
all_times=np.zeros(5)

vec=np.linspace(0,20,1000)
times=np.zeros(1000)
for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_ae=np.minimum(err_ae,np.mean(cross_val_score(GaussianProcessRegressor(alpha=alpha),AE_latent,AE_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[0]=err_ae
all_times[0]=np.mean(times)
    
for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_vae=np.minimum(err_vae,np.mean(cross_val_score(GaussianProcessRegressor(alpha=alpha),VAE_latent,VAE_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[1]=err_vae
all_times[1]=np.mean(times)

for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_aae=np.minimum(err_vae,np.mean(cross_val_score(GaussianProcessRegressor(alpha=alpha),VAE_latent,VAE_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[2]=err_aae
all_times[2]=np.mean(times)

for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_began=np.minimum(err_began,np.mean(cross_val_score(GaussianProcessRegressor(alpha=alpha),BEGAN_latent,BEGAN_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[3]=err_began
all_times[3]=np.mean(times)


for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_data=np.minimum(err_data,np.mean(cross_val_score(GaussianProcessRegressor(alpha=alpha),data_latent,data_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[4]=err_data
all_times[4]=np.mean(times)

print(all_err)
print(all_times)

np.save("err_gpr_drag.npy",all_err)
np.save("err_gpr_times.npy",all_times)