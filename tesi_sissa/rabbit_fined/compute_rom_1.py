
from ezyrb import Database,RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN, ReducedOrderModel, POD, AE, PODAE
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
np.random.seed(0)
from sklearn.utils.extmath import randomized_svd
import sys
names=["VAE",
       "AAE",
       "AE",
       "BEGAN",
       "AS",
       "data"
       ]



def l2_norm(x):
    return np.linalg.norm(x, axis=1)

def transform(x,R):
    return x.dot(R)

class L2_torch(nn.Module):
    def __init__(self,R):
        super().__init__()
        self.R=torch.tensor(R)
    
    def forward(self,x,y):
        tmp=x-y
        tmp=tmp@self.R
        return torch.linalg.norm(tmp)


class AdvancedRBF():
    
    def fit(self,x,y):
        self.flag=0
        if x.shape[1]>x.shape[0]:
            self.flag=1
            self.pca=PCA(n_components=x.shape[0]-1)
            self.pca.fit(x)
            x=self.pca.transform(x)
        self.rbf=RBF()
        self.rbf.fit(x,y)
    
    def predict(self,x):
        if self.flag==1:
            x=self.pca.transform(x)
        y=self.rbf.predict(x)
        return y


    
NUM_SAMPLES=100
NUM_TRAIN_SAMPLES=80
NUM_TEST=20
np.random.seed(0)

name=sys.argv[1]


train_error=np.zeros(3)
test_error=np.zeros(3)
#train_error=np.load("./rom_quantities/"+name+"_rom_err_train.npy")
#test_error=np.load("./rom_quantities/"+name+"_rom_err_test.npy",)


parameters=np.load("latent_variables/"+name+"_latent.npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1).astype(np.float32)

snapshot=np.load("physical_quantities/snaps_"+name+".npy").T[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1).astype(np.float32)
l=[]
for i in range(NUM_SAMPLES):
    if np.sum(np.isnan(snapshot[i]))<1:
        l.append(i)

parameters=parameters[l,:]  
snapshot=snapshot[l,:]

n_modes=3
U_1, S_1, V_1 = randomized_svd(snapshot, n_components=n_modes, random_state=0, n_oversamples=1)
R_1=V_1[:n_modes]
snapshot_red=snapshot.dot(R_1.T)
NUM_SAMPLES=len(snapshot)
train_index=np.random.choice(NUM_SAMPLES, NUM_TRAIN_SAMPLES, replace=False)
test_index=np.setdiff1d(np.arange(NUM_SAMPLES),train_index)
parameters_train=parameters[train_index]
parameters_test=parameters[test_index]
snapshot_train=snapshot[train_index]
snapshot_test=snapshot[test_index]
snapshot_train_red=snapshot_red[train_index]

approximations_names=["GPR","ANN","RBF"]


approximations = {
    'GPR': GPR(),
    'ANN': ANN([2000, 2000], nn.Tanh(), 1000,l2_regularization=0.10,lr=0.01, frequency_print=100),
    'RBF': AdvancedRBF(),
}



for approxname, approxclass in approximations.items():
    loss=L2_torch(R_1)
    j=list(approximations_names).index(approxname)
    approxclass.fit(parameters_train,snapshot_train_red)
    print(j)
    train_error[j]=np.mean(l2_norm(transform(approxclass.predict(parameters_train).reshape(NUM_TRAIN_SAMPLES,-1),R_1)-snapshot_train)/l2_norm(snapshot_train))
    test_error[j]=np.mean(l2_norm(transform(approxclass.predict(parameters_test).reshape(NUM_TEST,-1),R_1)-snapshot_test)/l2_norm(snapshot_test))

    


for j in range(len(approximations_names)):
    print("Training error of "+str(approximations_names[j])+"  is "+str(train_error[j]))
    print("Test error of "+str(approximations_names[j])+" is "+str(test_error[j]))

np.save("./rom_quantities/"+name+"_rom_err_train.npy",train_error)
np.save("./rom_quantities/"+name+"_rom_err_test.npy",test_error)
