import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

def get_kernel():
    s=gpytorch.kernels.RBFKernel()
    l=[1,2,5,10,20]
    for i in l:
        tmp=gpytorch.kernels.RBFKernel()
        tmp.lengthscale=i
        s=s+tmp
    return s

def mmd(X,Y):
    s=get_kernel()
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.mean(s(torch.tensor(X),torch.tensor(X)).to_dense().detach().numpy())+np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy())-2*np.mean(s(torch.tensor(X),torch.tensor(Y)).to_dense().detach().numpy())

names=["AE","VAE","AAE","BEGAN"]
db_t=["u","energy"]
approximations =  [
    'GPR',
    'ANN'
]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 20
})

var_tot=np.zeros(5)
mmd_tensor_tot=np.zeros(4)
rec_error_tot=np.zeros(4)
mmd_energy_tot=np.zeros(4)

train_energy_tot=np.zeros((6,2))
test_energy_tot=np.zeros((6,2))



for i in range(len(names)):
    name=names[i]
    moment_tensor_data=np.load("geometrical_quantities/moment_tensor_data.npy")
    moment_tensor_sampled=np.load("geometrical_quantities/moment_tensor_"+name+".npy")
    variance=np.load("nn_quantities/variance_"+name+".npy")
    variance_data=np.load("nn_quantities/variance_data.npy")
    error=np.load("nn_quantities/rel_error_"+name+".npy")
    energy_data=np.load("physical_quantities/energy_surf_data.npy").reshape(-1)
    energy_sampled=np.load("physical_quantities/energy_surf_"+name+".npy").reshape(-1)
    train_error_rom_sampled=np.load("./rom_quantities/"+name+"_rom_err_train.npy")
    test_error_rom_sampled=np.load("./rom_quantities/"+name+"_rom_err_test.npy")
    train_error_rom_data=np.load("./rom_quantities/data_rom_err_train.npy")
    test_error_rom_data=np.load("./rom_quantities/data_rom_err_test.npy")
    train_error_rom_as=np.load("./rom_quantities/AS_rom_err_train.npy")
    test_error_rom_as=np.load("./rom_quantities/AS_rom_err_test.npy")

    var_tot[0]=variance_data.item()
    mmd_tensor_tot[i]=mmd(moment_tensor_data.reshape(-1,np.prod(moment_tensor_data.shape[1:])),moment_tensor_sampled.reshape(-1,np.prod(moment_tensor_data.shape[1:])))
    var_tot[i+1]=variance.item()
    rec_error_tot[i]=error.item()
    mmd_energy_tot[i]=mmd(energy_data,energy_sampled)

    for j in range(len(approximations)):
        train_energy_tot[0,j]=train_error_rom_data[0,j]
        test_energy_tot[0,j]=test_error_rom_data[0,j]

        train_energy_tot[i+1,j]=train_error_rom_sampled[0,j]
        test_energy_tot[i+1,j]=test_error_rom_sampled[0,j]


        train_energy_tot[5,j]=train_error_rom_as[0,j]
        test_energy_tot[5,j]=test_error_rom_as[0,j]

    fig2,ax2=plt.subplots()
    ax2.set_title("Energy on surface of "+name)
    _=ax2.hist([energy_data,energy_sampled],8,label=['real','sampled'])
    ax2.grid(True,which='both')
    ax2.legend()
    fig2.savefig("./plots/Energy_surf_hist_"+name+"_fined.pdf")





plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 15
})



#Geometrical quantities
fig2,ax2=plt.subplots()
ax2.set_title("MMD between moment tensor of data and of GM")
ax2.plot(names,mmd_tensor_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/Moment_fined.pdf")
fig2,ax2=plt.subplots()
#Physical quantities
fig2,ax2=plt.subplots()
ax2.set_title("MMD between energy on surface of data and of GM")
ax2.plot(names,mmd_energy_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/Energy_surf_fined.pdf")
fig2,ax2=plt.subplots()
fig2,ax2=plt.subplots()
ax2.set_title("Rec error between data and GM")
ax2.plot(names,rec_error_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/rec_fined.pdf")
styles=['bo','gv','r.','y,']





fig2,ax2=plt.subplots()
ax2.set_title("Variance")
ax2.plot(["data"]+names,var_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/var_fined.pdf")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 13
})


names=["AE","VAE","AAE","BEGAN","AS"]

fig2,ax2=plt.subplots()
ax2.set_title("ROM u train error")



fig2,ax2=plt.subplots()
ax2.set_title("ROM u test error")


fig2,ax2=plt.subplots()
ax2.set_title("ROM energy on surf train error")



for j in range(len(approximations)):
    ax2.semilogy(["data"]+names,train_energy_tot[:,j],label=approximations[j])
ax2.grid(True,which='both')
ax2.legend()
fig2.savefig("./plots/train_energy_surf_fined.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("ROM energy on surf test error")

for j in range(len(approximations)):
    ax2.semilogy(["data"]+names,test_energy_tot[:,j],label=approximations[j])
ax2.grid(True,which='both') 
ax2.legend()
fig2.savefig("./plots/test_energy_surf_fined.pdf")
