from sklearn.tree import DecisionTreeRegressor
import numpy as np
from tqdm import trange
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from time import time
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer

def max_error(y_true, y_pred):
    return np.max(np.abs(y_true.reshape(-1,1) - y_pred.reshape(-1,1)))


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



all_drags=np.concatenate((AE_drag.reshape(1,-1,1),VAE_drag.reshape(1,-1,1),AAE_drag.reshape(1,-1,1),BEGAN_drag.reshape(1,-1,1),data_drag.reshape(1,-1,1)),axis=0)

all_latent={0:AE_latent,1:VAE_latent,2:AAE_latent,3:BEGAN_latent,4:data_latent}
#all_latent=np.concatenate((AE_latent.reshape(1,-1,1),VAE_latent.reshape(1,-1,1),AAE_latent.reshape(1,-1,1),BEGAN_latent.reshape(1,-1,1),data_latent.reshape(1,-1,1)),axis=0)

loo=LeaveOneOut()



all_err=np.zeros(5)
all_err_2=np.zeros(5)
all_times=np.zeros(5)
var_times=np.zeros(5)

vec=np.linspace(1,1000,1001).astype(np.int64)
times=np.zeros(1000)
for j in range(5):
    tmp=np.inf
    tmp_2=np.inf
    latent=all_latent[j]
    drags=all_drags[j]
    for i in trange(1000):
        alpha=vec[i]
        start=time()
        tmp=np.minimum(tmp,np.mean(cross_val_score(DecisionTreeRegressor(max_depth=alpha),latent, drags ,scoring=make_scorer(max_error),cv=loo)))
        tmp_2=np.minimum(tmp_2,np.sqrt(np.mean(cross_val_score(DecisionTreeRegressor(max_depth=alpha),latent, drags, scoring=make_scorer(max_error),cv=loo)**2)))
        end=time()
        times[i]=end-start  

    all_err[j]=tmp
    all_err_2[j]=tmp_2
    all_times[j]=np.mean(times)
    var_times[j]=np.var(times)


np.save("err_rf_drag.npy",all_err)
np.save("times_rf_drag.npy",all_times)
np.save("err_rf_drag_2.npy",all_err_2)
np.save("var_rf_drag.npy",var_times)