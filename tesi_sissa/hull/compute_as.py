from ezyrb import Database,RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN, ReducedOrderModel, POD, AE, PODAE
from  sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF as RBFKernel    
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from tqdm import trange
np.random.seed(0)
from athena.active import ActiveSubspaces
import GPy
from sklearn.utils.extmath import randomized_svd
NUM_SAMPLES=100
NUM_TRAIN_SAMPLES=75
parameters=np.load("latent_variables/data_latent.npy").reshape(600,-1)[:NUM_SAMPLES]
snapshot_1=np.load("physical_quantities/p_data.npy").reshape(NUM_SAMPLES,-1)

U_1, S_1, V_1 = randomized_svd(snapshot_1, n_components=100, random_state=0, n_oversamples=1)
R_1=V_1[:100]

snapshot_1=snapshot_1.dot(R_1.T)
d=parameters.shape[1]
k=GPy.kern.RBF(d,ARD=True,variance=1)
model=GPy.models.GPRegression(parameters,snapshot_1,k,normalizer=True)

model.optimize_restarts(10)
jacobian=model.predict_jacobian(parameters)[0].reshape(NUM_SAMPLES,100,86)
asub_1 = ActiveSubspaces(dim=10,method='exact',n_boot=50)

asub_1.fit(parameters,gradients=jacobian)
parameters_1=asub_1.transform(parameters)[0]
np.save("latent_variables/AS_latent.npy",parameters_1)
