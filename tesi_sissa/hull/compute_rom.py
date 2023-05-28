from ezyrb import Database,RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN, ReducedOrderModel, POD, AE, PODAE
from  sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF as RBFKernel    
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from tqdm import trange
np.random.seed(0)


names=["VAE",
       "AAE",
       "AE",
       "BEGAN",
       "AS",
       "data"
       ]

NUM_SAMPLES=100
NUM_TRAIN_SAMPLES=90
NUM_TEST=10
for name in names:
    np.random.seed(0)
    parameters=np.load("latent_variables/"+name+"_latent.npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    snapshot_1=np.load("physical_quantities/podded_p_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    snapshot_2=np.load("physical_quantities/podded_u_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)

    l=[]
    for i in range(NUM_SAMPLES):
        if np.sum(np.isnan(snapshot_1[i]))<1:
            l.append(i)

    parameters=parameters[l,:]  
    snapshot_1=snapshot_1[l,:]
    snapshot_2=snapshot_2[l,:]
    NUM_SAMPLES=len(snapshot_1)

    train_index=np.random.choice(NUM_SAMPLES, NUM_TRAIN_SAMPLES, replace=False)
    test_index=np.setdiff1d(np.arange(NUM_SAMPLES),train_index)
    parameters_train=parameters[train_index]
    parameters_test=parameters[test_index]
    snapshot_1_train=snapshot_1[train_index]
    snapshot_1_test=snapshot_1[test_index]
    snapshot_2_train=snapshot_2[train_index]
    snapshot_2_test=snapshot_2[test_index]



    db1=Database(parameters,snapshot_1)
    db2=Database(parameters,snapshot_2)

    db_t={"p": db1, "u":db2}

    train={"p":[parameters_train,snapshot_1_train],"u":[parameters_train,snapshot_2_train] }
    test={"p":[parameters_test,snapshot_1_test], "u":[parameters_test,snapshot_2_test] }



    approximations = {
        'GPR': GPR(),
        'ANN': ANN([2000, 2000], nn.Tanh(), 5000,l2_regularization=0.00,lr=0.01, frequency_print=1000),
    }


    train_error=np.zeros((2,2))
    test_error=np.zeros((2,2))
    
    for approxname, approxclass in approximations.items():
        podae=PODAE(POD('svd'),AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 50000, lr=0.001,frequency_print=10000))
        j=list(approximations.keys()).index(approxname)
        rom = ReducedOrderModel(db_t["p"], podae, approxclass)
        rom.fit()
        train_error[0,j]=np.linalg.norm(rom.predict(train["p"][0]).reshape(NUM_TRAIN_SAMPLES,-1)-train["p"][1])/np.linalg.norm(train["p"][1])
        test_error[0,j]=np.linalg.norm(rom.predict(test["p"][0]).reshape(NUM_TEST,-1)-test["p"][1])/np.linalg.norm(test["p"][1])
       
    
    approximations = {
        'GPR': GPR(),
        'ANN': ANN([2000, 2000], nn.Tanh(), 5000,l2_regularization=0.00,lr=0.01, frequency_print=1000),
    }


    for approxname, approxclass in approximations.items():
        podae=PODAE(POD('svd'),AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 50000, lr=0.001,frequency_print=10000))
        j=list(approximations.keys()).index(approxname)
        rom = ReducedOrderModel(db_t["u"], podae, approxclass)
        rom.fit()
        train_error[1,j]=np.linalg.norm(rom.predict(train["u"][0]).reshape(NUM_TRAIN_SAMPLES,-1)-train["u"][1])/np.linalg.norm(train["u"][1])
        test_error[1,j]=np.linalg.norm(rom.predict(test["u"][0]).reshape(NUM_TEST,-1)-test["u"][1])/np.linalg.norm(test["u"][1])



    approximations=list(approximations.keys())
    db_t=list(db_t.keys())
    for i in range(2):
        for j in range(len(approximations)):
            print("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error[i,j]))
            print("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error[i,j]))

    np.save("./rom_quantities/"+name+"_rom_err_train.npy",train_error)
    np.save("./rom_quantities/"+name+"_rom_err_test.npy",test_error)
