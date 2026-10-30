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




AE_latent=np.load("tesi_sissa/rabbit_coarse/latent_variables/AE_latent.npy")[:100]
VAE_latent=np.load("tesi_sissa/rabbit_coarse/latent_variables/VAE_latent.npy")[:100]
AAE_latent=np.load("tesi_sissa/rabbit_coarse/latent_variables/AAE_latent.npy")[:100]
BEGAN_latent=np.load("tesi_sissa/rabbit_coarse/latent_variables/BEGAN_latent.npy")[:100]
data_latent=np.load("tesi_sissa/rabbit_coarse/latent_variables/data_latent.npy")[:100].reshape(100,-1)
AE_drag=np.load("tesi_sissa/rabbit_coarse/physical_quantities/energy_AE.npy")[:100]
VAE_drag=np.load("tesi_sissa/rabbit_coarse/physical_quantities/energy_VAE.npy")[:100]
AAE_drag=np.load("tesi_sissa/rabbit_coarse/physical_quantities/energy_AAE.npy")[:100]
BEGAN_drag=np.load("tesi_sissa/rabbit_coarse/physical_quantities/energy_BEGAN.npy")[:100]
data_drag=np.load("tesi_sissa/rabbit_coarse/physical_quantities/energy_data.npy")[:100]




class RBFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='thin_plate_spline', epsilon=None, smoothing=0.0):
        self.kernel = kernel
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.rbf_interpolator_ = None

    def fit(self, X, y):
        """
        Fit the RBF interpolator model to the data.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns:
        self : returns an instance of self.
        """
        # Fit the RBF interpolator
        self.rbf_interpolator_ = RBFInterpolator(
            X, y, 
            kernel=self.kernel, 
            epsilon=self.epsilon, 
            smoothing=self.smoothing
        )
        return self

    def predict(self, X):
        """
        Predict using the RBF interpolator model.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        y_pred : array of shape (n_samples,)
            Predicted values.
        """
        # Check if the model has been fit
        if self.rbf_interpolator_ is None:
            raise RuntimeError("You must call `fit` before `predict`.")

        # Use the RBF interpolator to predict
        return self.rbf_interpolator_(X)



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
    err_ae=np.minimum(err_ae,np.mean(cross_val_score(RBFRegressor(smoothing=alpha),AE_latent,AE_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[0]=err_ae
all_times[0]=np.mean(times)
    
for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_vae=np.minimum(err_vae,np.mean(cross_val_score(RBFRegressor(smoothing=alpha),VAE_latent,VAE_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[1]=err_vae
all_times[1]=np.mean(times)

for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_aae=np.minimum(err_vae,np.mean(cross_val_score(RBFRegressor(smoothing=alpha),VAE_latent,VAE_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[2]=err_aae
all_times[2]=np.mean(times)

for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_began=np.minimum(err_began,np.mean(cross_val_score(RBFRegressor(smoothing=alpha),BEGAN_latent,BEGAN_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[3]=err_began
all_times[3]=np.mean(times)


for i in trange(1000):
    alpha=vec[i]
    start=time()
    err_data=np.minimum(err_data,np.mean(cross_val_score(RBFRegressor(smoothing=alpha),data_latent,data_drag,scoring=make_scorer(max_error),cv=loo)))
    end=time()
    times[i]=end-start  

all_err[4]=err_data
all_times[4]=np.mean(times)

np.save("err_rbf_energy.npy",all_err)
np.save("times_rbf_energy.npy",all_times)