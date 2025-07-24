import numpy as np
import matplotlib.pyplot as plt

name="drag"

if name=="energy":
    lb=5e-02
    ub=3e-01

if name=="drag":
    lb=1e-01
    ub=6e-01

lb_time=0.01
ub_time=1
gpr_energy=np.load("err_gpr_{}.npy".format(name))[1:]
rbf_energy=np.load("err_rbf_{}.npy".format(name))[1:]
rf_energy=np.load("err_rf_{}.npy".format(name))[1:]

times_gpr_energy=np.load("times_gpr_{}.npy".format(name))[1:]
times_rbf_energy=np.load("times_rbf_{}.npy".format(name))[1:]
times_rf_energy=np.load("times_rf_{}.npy".format(name))[1:]
var_gpr_energy=np.load("var_gpr_{}.npy".format(name))[1:]
var_rbf_energy=np.load("var_rbf_{}.npy".format(name))[1:]
var_rf_energy=np.load("var_rf_{}.npy".format(name))[1:]

gpr_energy_2=np.load("err_gpr_{}_2.npy".format(name))[1:]
rbf_energy_2=np.load("err_rbf_{}_2.npy".format(name))[1:]
rf_energy_2=np.load("err_rf_{}_2.npy".format(name))[1:]


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axes[0].semilogy(gpr_energy, markersize=10, color='r', marker='p', linestyle='none')
axes[0].set_title("GPR")
axes[0].grid(which='major')
axes[0].grid(which='minor', linestyle='dotted')
axes[0].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[0].set_ylim(lb, ub)
axes[1].semilogy(rbf_energy, markersize=10, color='r', marker='p', linestyle='none')
axes[1].set_title("RBF Interpolation")
axes[1].grid(which='major')
axes[1].grid(which='minor', linestyle='dotted')
axes[1].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[1].set_ylim(lb, ub)
axes[2].semilogy(rf_energy, markersize=10, color='r', marker='p', linestyle='none')
axes[2].set_title("Tree Interpolation")
axes[2].grid(which='major')
axes[2].grid(which='minor', linestyle='dotted')
axes[2].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[2].set_ylim(lb, ub)
axes[0].set_ylabel('mean_error')
fig.suptitle("Mean Error on the {}".format(name))
fig.savefig("error_{}.pdf".format(name))
plt.clf()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axes[0].semilogy(gpr_energy_2, markersize=10, color='r', marker='p', linestyle='none')
axes[0].set_title("GPR")
axes[0].grid(which='major')
axes[0].grid(which='minor', linestyle='dotted')
axes[0].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[0].set_ylim(lb, ub)
axes[1].semilogy(rbf_energy_2, markersize=10, color='r', marker='p', linestyle='none')
axes[1].set_title("RBF Interpolation")
axes[1].grid(which='major')
axes[1].grid(which='minor', linestyle='dotted')
axes[1].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[1].set_ylim(lb, ub)
axes[2].semilogy(rf_energy_2, markersize=10, color='r', marker='p', linestyle='none')
axes[2].set_title("Tree Interpolation")
axes[2].grid(which='major')
axes[2].grid(which='minor', linestyle='dotted')
axes[2].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[2].set_ylim(lb, ub)
axes[0].set_ylabel('max_error')
fig.suptitle("Max Error on the {}".format(name))
fig.savefig("error_{}_2.pdf".format(name))
plt.clf()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axes[0].semilogy(times_gpr_energy, markersize=10, color='r', marker='p', linestyle='none')
axes[0].fill_between(np.arange(4),times_gpr_energy-1.96*np.sqrt(var_gpr_energy),times_gpr_energy+1.96*np.sqrt(var_gpr_energy))
axes[0].set_title("GPR")
axes[0].grid(which='major')
axes[0].grid(which='minor', linestyle='dotted')
axes[0].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[0].set_ylim(lb_time, ub_time)
axes[1].semilogy(times_rbf_energy, markersize=10, color='r', marker='p', linestyle='none')
axes[1].set_title("RBF Interpolation")
axes[1].grid(which='major')
axes[1].grid(which='minor', linestyle='dotted')
axes[1].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[1].set_ylim(lb_time, ub_time)
axes[2].semilogy(times_rf_energy, markersize=10, color='r', marker='p', linestyle='none')
axes[2].set_title("Tree Interpolation")
axes[2].grid(which='major')
axes[2].grid(which='minor', linestyle='dotted')
axes[2].set_xticks(np.arange(4),["VAE","AAE","BEGAN","CFFD"])
#axes[2].set_ylim(lb_time, ub_time)
axes[0].set_ylabel('time')
fig.suptitle("Time of training and testing of the models on {}".format(name))

fig.savefig("time_{}.pdf".format(name))
