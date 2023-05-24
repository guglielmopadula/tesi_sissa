
import numpy as np
from tqdm import trange
np.random.seed(0)
from athena.active import ActiveSubspaces
import GPy

NUM_SAMPLES=300
NUM_TRAIN_SAMPLES=250
parameters=np.load("latent_variables/data_latent.npy").reshape(600,-1)[:300,]
snapshot_1=np.load("physical_quantities/energy_data.npy").reshape(NUM_SAMPLES,-1)

d=parameters.shape[1]
k=GPy.kern.RBF(d,ARD=True,variance=1)
model=GPy.models.GPRegression(parameters,snapshot_1,k,normalizer=True)
model.optimize_restarts(10)
jacobian=model.predict_jacobian(parameters)[0].reshape(NUM_SAMPLES,-1)

asub_1 = ActiveSubspaces(dim=5,method='exact')

asub_1.fit(parameters,gradients=jacobian)
parameters_1=asub_1.transform(parameters)[0]

np.save("latent_variables/AS_latent.npy",parameters_1)




