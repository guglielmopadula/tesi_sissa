from ezyrb import Database,RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN, ReducedOrderModel, POD, AE, PODAE
from  sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF as RBFKernel    
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from tqdm import trange
np.random.seed(0)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from athena import Normalizer
from tqdm import trange
import GPy
from sklearn.utils.extmath import randomized_svd

names=["VAE",
       #"AAE",
       #"AE",
       #"BEGAN",
       #"AS",
       #"data"
       ]

V=np.load("physical_quantities/VD.npy")

def l2_norm(x,R,V):
    x=x.dot(R)
    x = V.reshape(1, -1)*x
    return np.linalg.norm(x, axis=1)

NUM_SAMPLES=100
NUM_TRAIN_SAMPLES=99
NUM_TEST=1
for name in names:
    np.random.seed(0)
    parameters=np.load("latent_variables/"+name+"_latent.npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    snapshot_1=np.load("physical_quantities/p_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    snapshot_2=np.linalg.norm(np.load("physical_quantities/u_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,3,-1),axis=1)


    lb = np.min(parameters, axis=0)
    ub = np.max(parameters, axis=0)
    normalizer = Normalizer(lb, ub)
    parameters=normalizer.fit_transform(parameters)
  
    l=[]
    for i in range(NUM_SAMPLES):
        if np.sum(np.isnan(snapshot_1[i]))<1:
            l.append(i)

    parameters=parameters[l,:]  
    snapshot_1=snapshot_1[l,:]
    snapshot_2=snapshot_2[l,:]
    NUM_SAMPLES=len(snapshot_1)

    U_1, S_1, V_1 = randomized_svd(snapshot_1, n_components=100, random_state=0, n_oversamples=1)
    R_1=V_1[:100]
    print(R_1.shape)
    snapshot_1=snapshot_1.dot(R_1.T)
    U_2, S_2, V_2 = randomized_svd(snapshot_2, n_components=100, random_state=0, n_oversamples=1)
    R_2=V_2[:100]
    snapshot_2=snapshot_2.dot(R_2.T)


    train_index=np.random.choice(NUM_SAMPLES, NUM_TRAIN_SAMPLES, replace=False)
    test_index=np.setdiff1d(np.arange(NUM_SAMPLES),train_index)
    parameters_train=parameters[train_index]
    parameters_test=parameters[test_index]
    snapshot_1_train=snapshot_1[train_index]
    snapshot_1_test=snapshot_1[test_index]
    snapshot_2_train=snapshot_2[train_index]
    snapshot_2_test=snapshot_2[test_index]



    db1=Database(parameters_train,snapshot_1_train)
    db2=Database(parameters_train,snapshot_2_train)

    db_t={"p": db1, "u":db2}

    train={"p":[parameters_train,snapshot_1_train],"u":[parameters_train,snapshot_2_train] }
    test={"p":[parameters_test,snapshot_1_test], "u":[parameters_test,snapshot_2_test] }



    approximations = {
        'GPR': GaussianProcessRegressor(alpha=0.0001),
        #'ANN': ANN([2000, 2000], nn.Tanh(), 5000,l2_regularization=0.00,lr=0.01, frequency_print=1000),
    }


    train_error=np.zeros((2,2))
    test_error=np.zeros((2,2))
    
    for approxname, approxclass in approximations.items():
        #podae=PODAE(POD('svd'),AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 50000, lr=0.001,frequency_print=10000))
        j=list(approximations.keys()).index(approxname)
        #rom = ReducedOrderModel(db_t["p"], podae, approxclass)
        approxclass.fit(train["p"][0],train["p"][1])
        #print(train["p"][1].shape)
        train_error[0,j]=np.mean(l2_norm(approxclass.predict(train["p"][0]).reshape(NUM_TRAIN_SAMPLES,-1)-train["p"][1],R_1,V)/l2_norm(train["p"][1],R_1,V))
        test_error[0,j]=np.mean(l2_norm(approxclass.predict(test["p"][0]).reshape(NUM_TEST,-1)-test["p"][1],R_1,V)/l2_norm(test["p"][1],R_1,V))
       
    
    approximations = {
        'GPR': GaussianProcessRegressor(),
        #'ANN': ANN([2000, 2000], nn.Tanh(), 5000,l2_regularization=0.00,lr=0.01, frequency_print=1000),
    }


    for approxname, approxclass in approximations.items():
        #podae=PODAE(POD('svd'),AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 50000, lr=0.001,frequency_print=10000))
        j=list(approximations.keys()).index(approxname)
        #rom = ReducedOrderModel(db_t["u"], podae, approxclass)
        approxclass.fit(train["u"][0],train["u"][1])
        train_error[1,j]=np.mean(l2_norm(approxclass.predict(train["u"][0]).reshape(NUM_TRAIN_SAMPLES,-1)-train["u"][1],R_2,V)/l2_norm(train["u"][1],R_2,V))
        test_error[1,j]=np.mean(l2_norm(approxclass.predict(test["u"][0]).reshape(NUM_TEST,-1)-test["u"][1],R_2,V)/l2_norm(test["u"][1],R_2,V))


    approximations=list(approximations.keys())
    db_t=list(db_t.keys())
    for i in range(2):
        for j in range(len(approximations)):
            print("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error[i,j]))
            print("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error[i,j]))

    np.save("./rom_quantities/"+name+"_rom_err_train.npy",train_error)
    np.save("./rom_quantities/"+name+"_rom_err_test.npy",test_error)
