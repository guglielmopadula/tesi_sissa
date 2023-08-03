import numpy as np
gpr=np.zeros((6,2))
nn=np.zeros((6,2))
rbf=np.zeros((6,2))

archs = ["data","AS", "AAE", "AE", "VAE", "BEGAN"]

for i in range(len(archs)):
    train=np.load("rom_quantities/"+archs[i]+"_rom_err_train.npy")
    test=np.load("rom_quantities/"+archs[i]+"_rom_err_test.npy")
    gpr[i,0]=train.reshape(-1)[0]
    nn[i,0]=train.reshape(-1)[1]
    rbf[i,0]=train.reshape(-1)[2]
    gpr[i,1]=test.reshape(-1)[0]
    nn[i,1]=test.reshape(-1)[1]
    rbf[i,1]=test.reshape(-1)[2]

np.save("res_gpr.npy",gpr)
np.save("res_nn.npy",nn)
np.save("res_rbf.npy",rbf)