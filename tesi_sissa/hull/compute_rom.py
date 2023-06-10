
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



class EIM():

    def __init__(self,max_points=None):
        self.max_points=max_points

    def fit(self,x,y):
        if self.max_points==None:
            self.max_points=len(y)
        self.indices_mu=np.zeros(self.max_points,dtype=np.int32)
        self.indices_x=np.zeros(self.max_points,dtype=np.int32)
        self.indices_mu[0]=0
        index=0
        self.q=np.zeros((self.max_points,y.shape[1]))
        err=y[self.indices_mu[index]]
        self.indices_x[index]=np.argmax(np.abs(err))

        for i in trange(self.max_points):
            self.q[index]=err/err[self.indices_x[index]]
            if i!=self.max_points-1:
                res_indices=np.array(list(set(range(len(x))).difference(set(self.indices_mu))))
                res=np.zeros(len(res_indices))
                A=self.q[:index].T
                for i in np.arange(len(res_indices)):
                    alpha,_,_,_=np.linalg.lstsq(A,y[res_indices[i]])
                    res[i]=np.linalg.norm(A@alpha-y[res_indices[i]])
                index=index+1
                self.indices_mu[index]=res_indices[np.argmax(res)]
                alpha,_,_,_=np.linalg.lstsq(A,y[self.indices_mu[index]])
                err=y[self.indices_mu[index]]-A@alpha
                x_indices=np.array(list(set(range(self.max_points)).difference(set(self.indices_x))))
                self.indices_x[index]=x_indices[np.argmax(np.abs(err[x_indices]))]

        A=self.q.T
        alpha=np.zeros((self.max_points,self.max_points))
        for i in range(self.max_points):
            alpha[i],_,_,_=np.linalg.lstsq(A,y[i])

        self.rbf=AdvancedRBF()
        self.rbf.fit(x[self.indices_x],alpha)

    def predict(self,mu):
        return self.rbf.predict(mu).dot(self.q)
    

class DEIM():

    def __init__(self,max_points=None):
        self.max_points=max_points

    def fit(self,x,y):
        if self.max_points==None:
            self.max_points=len(y)
        self.indices_mu=np.zeros(self.max_points,dtype=np.int32)
        self.indices_x=np.zeros(self.max_points,dtype=np.int32)
        self.indices_mu[0]=0
        index=0
        self.q=np.zeros((self.max_points,y.shape[1]))
        err=y[self.indices_mu[index]]
        self.indices_x[index]=np.argmax(np.abs(err))
        _,_,y_train=randomized_svd(y,n_components=3)

        for i in trange(self.max_points):
            self.q[index]=err/err[self.indices_x[index]]
            if i!=self.max_points-1:
                res_indices=np.array(list(set(range(self.max_points)).difference(set(self.indices_mu))))
                res=np.zeros(len(res_indices))
                A=self.q[:index].T
                for i in np.arange(len(res_indices)):
                    alpha,_,_,_=np.linalg.lstsq(A,y_train[res_indices[i]])
                    res[i]=np.linalg.norm(A@alpha-y_train[res_indices[i]])
                index=index+1
                self.indices_mu[index]=res_indices[np.argmax(res)]
                alpha,_,_,_=np.linalg.lstsq(A,y_train[self.indices_mu[index]])
                err=y[self.indices_mu[index]]-A@alpha
                x_indices=np.array(list(set(range(self.max_points)).difference(set(self.indices_x))))
                self.indices_x[index]=x_indices[np.argmax(np.abs(err[x_indices]))]

        A=self.q.T
        alpha=np.zeros((self.max_points,self.max_points))
        for i in range(self.max_points):
            alpha[i],_,_,_=np.linalg.lstsq(A,y[i])

        self.rbf=AdvancedRBF()
        self.rbf.fit(x[self.indices_x],alpha)

    def predict(self,mu):
        return self.rbf.predict(mu).dot(self.q)
    

V=np.load("physical_quantities/VD.npy").astype(np.float32)

def l2_norm(x,R,V):
    x=x.dot(R)
    x = V.reshape(1, -1)*x
    return np.linalg.norm(x, axis=1)

class L2_torch(nn.Module):
    def __init__(self,V,R):
        super().__init__()
        self.V=torch.tensor(V)
        self.R=torch.tensor(R)
    
    def forward(self,x,y):
        tmp=x-y
        tmp=tmp@self.R
        tmp = self.V.reshape(1, -1)*tmp
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

train_error=np.load("./rom_quantities/"+name+"_rom_err_train.npy")
test_error=np.load("./rom_quantities/"+name+"_rom_err_test.npy",)


parameters=np.load("latent_variables/"+name+"_latent.npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1).astype(np.float32)
snapshot_1=np.load("physical_quantities/p_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,-1).astype(np.float32)
snapshot_2=np.linalg.norm(np.load("physical_quantities/u_"+name+".npy")[:NUM_SAMPLES].reshape(NUM_SAMPLES,3,-1),axis=1).reshape(NUM_SAMPLES,-1).astype(np.float32)
l=[]
for i in range(NUM_SAMPLES):
    if np.sum(np.isnan(snapshot_1[i]))<1:
        l.append(i)

parameters=parameters[l,:]  
snapshot_1=snapshot_1[l,:]
snapshot_2=snapshot_2[l,:]

n_modes=3

U_1, S_1, V_1 = randomized_svd(snapshot_1, n_components=n_modes, random_state=0, n_oversamples=1)

R_1=V_1[:n_modes]
snapshot_1=snapshot_1.dot(R_1.T)


U_2, S_2, V_2 = randomized_svd(snapshot_2, n_components=n_modes, random_state=0, n_oversamples=1)
R_2=V_2[:n_modes]
snapshot_2=snapshot_2.dot(R_2.T)

print(snapshot_1.dtype)

NUM_SAMPLES=len(snapshot_1)

train_index=np.random.choice(NUM_SAMPLES, NUM_TRAIN_SAMPLES, replace=False)
test_index=np.setdiff1d(np.arange(NUM_SAMPLES),train_index)
parameters_train=parameters[train_index]
parameters_test=parameters[test_index]
snapshot_1_train=snapshot_1[train_index]
snapshot_1_test=snapshot_1[test_index]
snapshot_2_train=snapshot_2[train_index]
snapshot_2_test=snapshot_2[test_index]

approximations_names=["GPR","ANN","RBF","EIM","DEIM"]


db1=Database(parameters_train,snapshot_1_train)
db2=Database(parameters_train,snapshot_2_train)

db_t={"p": db1, "u":db2}

train={"p":[parameters_train,snapshot_1_train],"u":[parameters_train,snapshot_2_train] }
test={"p":[parameters_test,snapshot_1_test], "u":[parameters_test,snapshot_2_test] }



approximations = {
    #'GPR': GPR(),
    #'ANN': ANN([2000, 2000], nn.Tanh(), 5000,l2_regularization=0.00,lr=0.01, frequency_print=1000),
    #'RBF': AdvancedRBF(),
    #'EIM': EIM(3),
    'DEIM': DEIM(3),
}



for approxname, approxclass in approximations.items():
    loss=L2_torch(V,R_1)
    j=list(approximations_names).index(approxname)
    approxclass.fit(train["p"][0],train["p"][1])
    print(j)
    train_error[0,j]=np.mean(l2_norm(approxclass.predict(train["p"][0]).reshape(NUM_TRAIN_SAMPLES,-1)-train["p"][1],R_1,V)/l2_norm(train["p"][1],R_1,V))
    test_error[0,j]=np.mean(l2_norm(approxclass.predict(test["p"][0]).reshape(NUM_TEST,-1)-test["p"][1],R_1,V)/l2_norm(test["p"][1],R_1,V))

    

approximations = {
    #'GPR': GPR(),
    #'ANN': ANN([2000, 2000], nn.Tanh(), 5000,l2_regularization=0.00,lr=0.01, frequency_print=1000),
    #'RBF': AdvancedRBF(),
    #'EIM': EIM(3),
   'DEIM': DEIM(3),
}


for approxname, approxclass in approximations.items():
    approxclass.fit(train["u"][0],train["u"][1])
    j=list(approximations_names).index(approxname)
    print(j)
    train_error[1,j]=np.mean(l2_norm(approxclass.predict(train["u"][0]).reshape(NUM_TRAIN_SAMPLES,-1)-train["u"][1],R_2,V)/l2_norm(train["u"][1],R_2,V))
    test_error[1,j]=np.mean(l2_norm(approxclass.predict(test["u"][0]).reshape(NUM_TEST,-1)-test["u"][1],R_2,V)/l2_norm(test["u"][1],R_2,V))

 


db_t=list(db_t.keys())
for i in range(2):
    for j in range(len(approximations_names)):
        print("Training error of "+str(approximations_names[j])+" over " + str(db_t[i]) +" is "+str(train_error[i,j]))
        print("Test error of "+str(approximations_names[j])+" over " + str(db_t[i]) +" is "+str(test_error[i,j]))

#np.save("./rom_quantities/"+name+"_rom_err_train.npy",train_error)
#np.save("./rom_quantities/"+name+"_rom_err_test.npy",test_error)
