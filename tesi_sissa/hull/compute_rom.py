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
for name in names:
    np.random.seed(0)
    parameters=np.load("latent_variables/"+name+"_latent.npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    snapshot_1=np.load("physical_quantities/mean_square_pressure_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)
    snapshot_2=np.load("physical_quantities/mean_velocity_magnitude_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1)

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

    db_t={"mean_square_pressure": db1, "mean_velocity_magnitude":db2}

    train={"mean_square_pressure":[parameters_train,snapshot_1_train],"mean_velocity_magnitude":[parameters_train,snapshot_2_train] }
    test={"mean_square_pressure":[parameters_test,snapshot_1_test], "mean_velocity_magnitude":[parameters_test,snapshot_2_test] }


    podae=PODAE(POD('svd'),AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 5000))

    approximations = {
        'GPR': GPR(),
    #    'KNeighbors': KNeighborsRegressor(),
    #    'ANN': ANN([2000, 2000], nn.Tanh(), 5000,l2_regularization=0.00,lr=0.1),
    #s    'RBF': RBF(),
    }


    train_error=np.zeros((2,4))
    test_error=np.zeros((2,4))

    for approxname, approxclass in approximations.items():
        j=list(approximations.keys()).index(approxname)
        approxclass.fit(train["mean_square_pressure"][0],train["mean_square_pressure"][1],kern=ConstantKernel()*RBFKernel())
        train_error[0,j]=np.linalg.norm(approxclass.predict(train["mean_square_pressure"][0]).reshape(-1,1)-train["mean_square_pressure"][1])/np.linalg.norm(train["mean_square_pressure"][1])
        test_error[0,j]=np.linalg.norm(approxclass.predict(test["mean_square_pressure"][0]).reshape(-1,1)-test["mean_square_pressure"][1])/np.linalg.norm(test["mean_square_pressure"][1])
   
    approximations = {
        'GPR': GPR(),
       # 'KNeighbors': KNeighborsRegressor(),
       # 'ANN': ANN([2000, 2000], nn.Tanh(), 1000,l2_regularization=0.03,lr=0.001),
       # 'RBF': RBF(),
    }


    for approxname, approxclass in approximations.items():
        j=list(approximations.keys()).index(approxname)
        approxclass.fit(train["mean_velocity_magnitude"][0],train["mean_velocity_magnitude"][1])
        train_error[1,j]=np.linalg.norm(approxclass.predict(train["mean_velocity_magnitude"][0]).reshape(-1,1)-train["mean_velocity_magnitude"][1])/np.linalg.norm(train["mean_velocity_magnitude"][1])
        test_error[1,j]=np.linalg.norm(approxclass.predict(test["mean_velocity_magnitude"][0]).reshape(-1,1)-test["mean_velocity_magnitude"][1])/np.linalg.norm(test["mean_velocity_magnitude"][1])



    approximations=list(approximations.keys())
    db_t=list(db_t.keys())
    #f = open("./rom/graphs_txt/"+name+"_.txt", "a")
    for i in range(1):
        for j in range(len(approximations)):
            #print("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error[i,j]))
            print("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error[i,j]))

    np.save("./rom_quantities/"+name+"_rom_err_train.npy",train_error)
    np.save("./rom_quantities/"+name+"_rom_err_test.npy",test_error)
