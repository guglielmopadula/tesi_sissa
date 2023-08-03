import numpy as np
gpr_u=np.zeros((6,2))
nn_u=np.zeros((6,2))
rbf_u=np.zeros((6,2))

archs = ["data","AS", "AAE", "AE", "VAE", "BEGAN"]

for i in range(len(archs)):
    train=np.load("rom_quantities/"+archs[i]+"_rom_err_train.npy")
    test=np.load("rom_quantities/"+archs[i]+"_rom_err_test.npy")
    gpr_u[i,0]=train.T[0,0]
    nn_u[i,0]=train.T[1,0]
    rbf_u[i,0]=train.T[2,0]
    gpr_u[i,1]=test.T[0,0]
    nn_u[i,1]=test.T[1,0]
    rbf_u[i,1]=test.T[2,0]

np.save("res_gpr_u.npy",gpr_u)
np.save("res_nn_u.npy",nn_u)
np.save("res_rbf_u.npy",rbf_u)



gpr_p=np.zeros((6,2))
nn_p=np.zeros((6,2))
rbf_p=np.zeros((6,2))

archs = ["data","AS", "AAE", "AE", "VAE", "BEGAN"]

for i in range(len(archs)):
    train=np.load("rom_quantities/"+archs[i]+"_rom_err_train.npy")
    test=np.load("rom_quantities/"+archs[i]+"_rom_err_test.npy")
    gpr_p[i,0]=train.T[0,1]
    nn_p[i,0]=train.T[1,1]
    rbf_p[i,0]=train.T[2,1]
    gpr_p[i,1]=test.T[0,1]
    nn_p[i,1]=test.T[1,1]
    rbf_p[i,1]=test.T[2,1]

np.save("res_gpr_p.npy",gpr_p)
np.save("res_nn_p.npy",nn_p)
np.save("res_rbf_p.npy",rbf_p)