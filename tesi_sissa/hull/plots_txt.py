import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

def get_kernel(m=1):
    s=gpytorch.kernels.RBFKernel()
    l=[1,2,5,10,20]
    for i in l:
        tmp=gpytorch.kernels.RBFKernel()
        tmp.lengthscale=i*(m**0.5)
        s=s+tmp
    return s

def mmd(X,Y):
    s=get_kernel()
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.mean(s(torch.tensor(X),torch.tensor(X)).to_dense().detach().numpy())+np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy())-2*np.mean(s(torch.tensor(X),torch.tensor(Y)).to_dense().detach().numpy())


def relmmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    Z=np.concatenate((X,Y))
    s=get_kernel(X.shape[1])
    X=(X-np.min(Z,axis=0))/(np.max(Z,axis=0)-np.min(Z,axis=0))
    Y=(Y-np.min(Y,axis=0))/(np.max(Z,axis=0)-np.min(Z,axis=0))
    return (np.mean(s(torch.tensor(X),torch.tensor(X)).to_dense().detach().numpy())+np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy())-2*np.mean(s(torch.tensor(X),torch.tensor(Y)).to_dense().detach().numpy()))/np.mean(s(torch.tensor(X),torch.tensor(X)).to_dense().detach().numpy())

names=["AE","VAE","AAE","BEGAN"]
db_t=["u","energy"]
approximations =  [
    'POD-GPR',
    'POD-ANN',
    'POD-RBF',
    #'EIM',
    #'DEIM',
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
kid_tot=np.zeros(4)
mmd_drag_tot=np.zeros(4)
mmd_momz_tot=np.zeros(4)

train_u_tot=np.zeros((6,5))
test_u_tot=np.zeros((6,5))

train_p_tot=np.zeros((6,5))
test_p_tot=np.zeros((6,5))


for i in range(len(names)):
    name=names[i]
    moment_tensor_data=np.load("geometrical_quantities/moment_tensor_data.npy")
    moment_tensor_sampled=np.load("geometrical_quantities/moment_tensor_"+name+".npy")
    variance=np.load("nn_quantities/variance_"+name+".npy")
    variance_data=np.load("nn_quantities/variance_data.npy")
    area_data=np.load("geometrical_quantities/area_data.npy")
    area_sampled=np.load("geometrical_quantities/area_"+name+".npy")
    error=np.load("nn_quantities/rel_error_"+name+".npy")
    kid=np.load("nn_quantities/kid_"+name+".npy")
    drag_data=np.load("physical_quantities/drag_data.npy").reshape(-1)
    drag_sampled=np.load("physical_quantities/drag_"+name+".npy").reshape(-1)
    momz_data=np.load("physical_quantities/momz_data.npy").reshape(-1)
    momz_sampled=np.load("physical_quantities/momz_"+name+".npy").reshape(-1)

    train_error_rom_sampled=np.load("./rom_quantities/"+name+"_rom_err_train.npy")
    test_error_rom_sampled=np.load("./rom_quantities/"+name+"_rom_err_test.npy")
    train_error_rom_data=np.load("./rom_quantities/data_rom_err_train.npy")
    test_error_rom_data=np.load("./rom_quantities/data_rom_err_test.npy")
    train_error_rom_as=np.load("./rom_quantities/AS_rom_err_train.npy")
    test_error_rom_as=np.load("./rom_quantities/AS_rom_err_test.npy")

    var_tot[0]=variance_data.item()
    mmd_tensor_tot[i]=relmmd(moment_tensor_data.reshape(-1,np.prod(moment_tensor_data.shape[1:])),moment_tensor_sampled.reshape(-1,np.prod(moment_tensor_data.shape[1:])))
    var_tot[i+1]=variance.item()
    rec_error_tot[i]=error.item()
    kid_tot[i]=kid
    mmd_drag_tot[i]=relmmd(drag_data,drag_sampled)
    mmd_momz_tot[i]=relmmd(momz_data,momz_sampled)

    mmd_area_tot[i]=relmmd(area_data,area_sampled)
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
    _=ax2.hist(area_data,8,label='real',histtype='step',linestyle='solid',density=True)
    _=ax2.hist(area_sampled,8,label='sampled',histtype='step',linestyle='dotted',density=True)

    ax2.grid(True,which='both')
    ax2.legend()
    fig2.savefig("./plots/Area_hist_"+name+"_hull.pdf")
    fig2,ax2=plt.subplots()
    ax2.set_title("Drag of "+name)
    _=ax2.hist(drag_data,8,label='real',histtype='step',linestyle='solid',density=True)
    _=ax2.hist(drag_sampled,8,label='sampled',histtype='step',linestyle='dotted',density=True)
    ax2.grid(True,which='both')
    ax2.legend()
    fig2.savefig("./plots/Drag_hist_"+name+"_hull.pdf")
    fig2,ax2=plt.subplots()
    ax2.set_title("Z angular moment of "+name)
    _=ax2.hist(momz_data,8,label='real',histtype='step',linestyle='solid',density=True)
    _=ax2.hist(momz_sampled,8,label='sampled',histtype='step',linestyle='dotted',density=True)
    ax2.grid(True,which='both')
    ax2.legend()
    fig2.savefig("./plots/Momz_hist_"+name+"_hull.pdf")






plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 15
})



#Geometrical quantities
fig2,ax2=plt.subplots()
ax2.set_title("RelMMD between moment tensor of data and of GM")
ax2.plot(names,mmd_tensor_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/Moment_hull.pdf")
fig2,ax2=plt.subplots()
ax2.grid(True,which='both')
ax2.set_title("RelMMD between area of data and of GM")
ax2.plot(names,mmd_area_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/Area_hull.pdf")
#Physical quantities
fig2,ax2=plt.subplots()
ax2.set_title("RelMMD between drag of data and of GM")
ax2.plot(names,mmd_drag_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/Drag_hull.pdf")
fig2,ax2=plt.subplots()
ax2.set_title("RelMMD between z angular moment of data and of GM")
ax2.plot(names,mmd_momz_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/Momz_hull.pdf")

fig2,ax2=plt.subplots()
ax2.set_title("Rec error between data and GM")
ax2.plot(names,rec_error_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/rec_hull.pdf")
styles=['bo','gv','r.','y,']


fig2,ax2=plt.subplots()
ax2.set_title("KID between data and GM")
ax2.plot(names,kid_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/kid_hull.pdf")



fig2,ax2=plt.subplots()
ax2.set_title("Variance")
ax2.plot(["data"]+names,var_tot)
ax2.grid(True,which='both')
fig2.savefig("./plots/var_hull.pdf")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': ['Computer Modern'],
    "font.size": 13
})


names=["AE","VAE","AAE","BEGAN","AS"]

fig2,ax2=plt.subplots()
ax2.set_title("ROM pressure train error")


style=['solid','dotted','dashed','dashdot','dotted']

for j in range(len(approximations)):
    ax2.semilogy(["data"]+names,train_p_tot[:,j],label=approximations[j],linestyle=style[j])

ax2.grid(True,which='both')
ax2.legend()
fig2.savefig("./plots/train_pressure_hull.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("ROM pressure test error")

for j in range(len(approximations)):
    ax2.semilogy(["data"]+names,test_p_tot[:,j],label=approximations[j],linestyle=style[j])

ax2.grid(True,which='both')
ax2.legend()
fig2.savefig("./plots/test_pressure_hull.pdf")

fig2,ax2=plt.subplots()
ax2.set_title("ROM velocity magnitude train error")


for j in range(len(approximations)):
    ax2.semilogy(["data"]+names,train_u_tot[:,j],label=approximations[j],linestyle=style[j])
ax2.grid(True,which='both')
ax2.legend()
fig2.savefig("./plots/train_velocity_hull.pdf")


fig2,ax2=plt.subplots()
ax2.set_title("ROM velocity magnitude test error")

for j in range(len(approximations)):
    ax2.semilogy(["data"]+names,test_u_tot[:,j],label=approximations[j],linestyle=style[j])
    
    
ax2.grid(True,which='both')
ax2.legend()
fig2.savefig("./plots/test_velocity_hull.pdf")