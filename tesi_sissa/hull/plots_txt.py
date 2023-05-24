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
  #  'KNeighbors',
  #  'ANN',
  #  'RBF'

]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 20
})

var_tot=np.zeros(5)
mmd_tensor_tot=np.zeros(4)
mmd_area_tot=np.zeros(4)
rec_error_tot=np.zeros(4)
mmd_u_tot=np.zeros(4)
mmd_p_tot=np.zeros(4)

train_u_tot=np.zeros((6,4))
test_u_tot=np.zeros((6,4))

train_p_tot=np.zeros((6,4))
test_p_tot=np.zeros((6,4))


for i in range(len(names)):
    name=names[i]
    moment_tensor_data=np.load("geometrical_quantities/moment_tensor_data.npy")
    moment_tensor_sampled=np.load("geometrical_quantities/moment_tensor_"+name+".npy")
    variance=np.load("nn_quantities/variance_"+name+".npy")
    variance_data=np.load("nn_quantities/variance_data.npy")
    area_data=np.load("geometrical_quantities/area_data.npy")
    area_sampled=np.load("geometrical_quantities/area_"+name+".npy")
    error=np.load("nn_quantities/rel_error_"+name+".npy")
    u_data=np.load("physical_quantities/mean_velocity_magnitude_data.npy").reshape(-1)
    u_sampled=np.load("physical_quantities/mean_velocity_magnitude_"+name+".npy").reshape(-1)
    p_data=np.load("physical_quantities/mean_square_pressure_data.npy").reshape(-1)
    p_sampled=np.load("physical_quantities/mean_square_pressure_"+name+".npy").reshape(-1)

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
    mmd_u_tot[i]=mmd(u_data,u_sampled)
    mmd_u_tot[i]=mmd(u_data,u_sampled)

    mmd_area_tot[i]=mmd(area_data,area_sampled)
    for j in range(len(approximations)):
        train_p_tot[0,j]=train_error_rom_data[0,j]
        test_p_tot[0,j]=test_error_rom_data[0,j]
        train_p_tot[i+1,j]=train_error_rom_sampled[0,j]
        test_p_tot[i+1,j]=test_error_rom_sampled[0,j]
        train_p_tot[5,j]=train_error_rom_as[0,j]
        test_p_tot[5,j]=test_error_rom_as[0,j]

        train_u_tot[0,j]=train_error_rom_data[1,j]
        test_u_tot[0,j]=test_error_rom_data[1,j]
        train_u_tot[i+1,j]=train_error_rom_sampled[1,j]
        test_u_tot[i+1,j]=test_error_rom_sampled[1,j]
        train_u_tot[5,j]=train_error_rom_as[1,j]
        test_u_tot[5,j]=test_error_rom_as[1,j]
    fig2,ax2=plt.subplots()
    ax2.set_title("Area of "+name)
    _=ax2.hist([area_data,area_sampled],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./plots/Area_hist_"+name+".pdf")
    fig2,ax2=plt.subplots()
    ax2.set_title("Mean Square Pressure of "+name)
    _=ax2.hist([p_data,p_sampled],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./plots/Pressure_hist_"+name+".pdf")
    fig2,ax2=plt.subplots()
    ax2.set_title("Mean Velocity Magnitude Pressure of "+name)
    _=ax2.hist([u_data,u_sampled],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./plots/Velocity_hist_"+name+".pdf")






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
fig2.savefig("./plots/Moment.pdf")
fig2,ax2=plt.subplots()
ax2.set_title("MMD between area of data and of GM")
ax2.plot(names,mmd_area_tot)
fig2.savefig("./plots/Area.pdf")
#Physical quantities
fig2,ax2=plt.subplots()
ax2.set_title("MMD between mean square pressure of data and of GM")
ax2.plot(names,mmd_p_tot)
fig2.savefig("./plots/Pressure.pdf")
fig2,ax2=plt.subplots()
ax2.set_title("MMD between mean velocity magnitude of data and of GM")
ax2.plot(names,mmd_u_tot)
fig2.savefig("./plots/Velocity.pdf")

fig2,ax2=plt.subplots()
fig2,ax2=plt.subplots()
ax2.set_title("Rec error between data and GM")
ax2.plot(names,rec_error_tot)
fig2.savefig("./plots/rec.pdf")
styles=['bo','gv','r.','y,']





fig2,ax2=plt.subplots()
ax2.set_title("Variance")
ax2.plot(["data"]+names,var_tot)
fig2.savefig("./plots/var.pdf")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 13
})


names=["AE","VAE","AAE","BEGAN","AS"]

fig2,ax2=plt.subplots()
ax2.set_title("ROM mean square pressure train error")



for j in range(len(approximations)):
    if approximations[j]=="RBF":
        y_lim=ax2.get_ylim()
        ax2.plot(["data"]+names,train_p_tot[:,j],label=approximations[j])
        ax2.set_ylim(y_lim)
    else:
        ax2.plot(["data"]+names,train_p_tot[:,j],label=approximations[j])

    
ax2.legend()
fig2.savefig("./plots/train_pressure.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("ROM mean square pressure test error")

for j in range(len(approximations)):
    if approximations[j]=="RBF":
        y_lim=ax2.get_ylim()
        ax2.plot(["data"]+names,test_p_tot[:,j],label=approximations[j])
        ax2.set_ylim(y_lim)
    else:
        ax2.plot(["data"]+names,test_p_tot[:,j],label=approximations[j])

    
ax2.legend()
fig2.savefig("./plots/test_pressure.pdf")

fig2,ax2=plt.subplots()
ax2.set_title("ROM mean velocity magnitude train error")



for j in range(len(approximations)):
    if approximations[j]=="RBF":
        y_lim=ax2.get_ylim()
        ax2.plot(["data"]+names,train_u_tot[:,j],label=approximations[j])
        ax2.set_ylim(y_lim)
    else:
        ax2.plot(["data"]+names,train_u_tot[:,j],label=approximations[j])

    
ax2.legend()
fig2.savefig("./plots/train_velocity.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("ROM mean velocity magnitude test error")

for j in range(len(approximations)):
    if approximations[j]=="RBF":
        y_lim=ax2.get_ylim()
        ax2.plot(["data"]+names,test_u_tot[:,j],label=approximations[j])
        ax2.set_ylim(y_lim)
    else:
        ax2.plot(["data"]+names,test_u_tot[:,j],label=approximations[j])

    
    
ax2.legend()
fig2.savefig("./plots/test_velocity.pdf")