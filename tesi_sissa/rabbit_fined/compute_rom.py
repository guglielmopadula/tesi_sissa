from ezyrb import Database,RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN, ReducedOrderModel, POD, AE, PODAE
from  sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF as RBFKernel    
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from tqdm import trange
np.random.seed(0)
from sklearn.preprocessing import StandardScaler

names=["VAE",
       "AAE",
       "AE",
       "BEGAN",
       "AS",
       "data",
       ]


class AdvancedRBF():
    
    def fit(self,x,y):
        self.pca=PCA()
        self.pca.fit(x)
        self.reduced_dim=np.argmin(np.linalg.norm(self.pca.explained_variance_-0.999))+1
        self.pca=PCA(n_components=self.reduced_dim)
        self.pca.fit(x)
        x=self.pca.transform(x)
        self.rbf=RBF()
        self.rbf.fit(x,y)
    
    def predict(self,x):
        x=self.pca.transform(x)
        y=self.rbf.predict(x)
        return y

NUM_SAMPLES=200
NUM_TRAIN_SAMPLES=150
for name in names:
    np.random.seed(0)
    parameters=np.load("latent_variables/"+name+"_latent.npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    snapshot_1=np.load("physical_quantities/energy_surf_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    l=[]
    for i in range(NUM_SAMPLES):
        if np.sum(np.isnan(snapshot_1[i]))<1:
            l.append(i)


    parameters=parameters[l,:]  
    snapshot_1=snapshot_1[l,:]
    NUM_SAMPLES=len(snapshot_1)

    train_index=np.random.choice(NUM_SAMPLES, NUM_TRAIN_SAMPLES, replace=False)
    test_index=np.setdiff1d(np.arange(NUM_SAMPLES),train_index)
    parameters_train=parameters[train_index]
    parameters_test=parameters[test_index]
    snapshot_1_train=snapshot_1[train_index]
    snapshot_1_test=snapshot_1[test_index]



    db1=Database(parameters,snapshot_1)

    db_t={"energy": db1}

    train={"energy":[parameters_train,snapshot_1_train] }
    test={"energy":[parameters_test,snapshot_1_test] }


    podae=PODAE(POD('svd'),AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 5000))

    approximations = {
        'GPR': GPR(),
        'ANN': ANN([2000, 2000], nn.Tanh(), 1000,l2_regularization=0.03,lr=0.001),
        'RBF': AdvancedRBF()
    }


    train_error=np.zeros((1,3))
    test_error=np.zeros((1,3))

    for approxname, approxclass in approximations.items():
        j=list(approximations.keys()).index(approxname)
        approxclass.fit(train["energy"][0],train["energy"][1])
        train_error[0,j]=np.linalg.norm(approxclass.predict(train["energy"][0]).reshape(-1,1)-train["energy"][1])/np.linalg.norm(train["energy"][1])
        test_error[0,j]=np.linalg.norm(approxclass.predict(test["energy"][0]).reshape(-1,1)-test["energy"][1])/np.linalg.norm(test["energy"][1])



    approximations=list(approximations.keys())
    db_t=list(db_t.keys())
    #f = open("./rom/graphs_txt/"+name+"_.txt", "a")
    for i in range(len(db_t)):
        for j in range(len(approximations)):
            print("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error[i,j]))
            print("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error[i,j]))

    np.save("./rom_quantities/"+name+"_rom_err_train.npy",train_error)
    np.save("./rom_quantities/"+name+"_rom_err_test.npy",test_error)
