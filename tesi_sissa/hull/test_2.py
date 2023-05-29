
import numpy as np
from athena import ActiveSubspaces, Normalizer, NonlinearLevelSet, local_linear_gradients, rrmse
from sklearn.utils.extmath import randomized_svd
for name in ["AE","VAE","data","BEGAN","AAE"]:
    out = np.linalg.norm(np.load("./physical_quantities/u_"+name+".npy").reshape(100, 3, -1), axis=1, keepdims=False)
    r_dim=100 #100
    U, S, V = randomized_svd(out, n_components=r_dim, random_state=0, n_oversamples=1)
    R = V[:r_dim]
    recErrTrain = np.linalg.norm(out-out.dot(R.T).dot(R))/np.linalg.norm(out)
    print("train, test: ", R.shape, np.max(recErrTrain))
    reduced_coordinates = out.dot(R.T)
    np.save("svd_u_"+name+".npy",reduced_coordinates)

