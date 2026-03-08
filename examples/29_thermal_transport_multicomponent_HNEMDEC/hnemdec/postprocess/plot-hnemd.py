import numpy as np
import matplotlib.pyplot as plt

def get_ste(y):
    std = np.std(y)
    n = len(y)
    ste = std/(n**0.5)
    return ste

def get_running(data):
    running = np.cumsum(data)
    count = np.arange(1,1+len(data))
    running_ave = running/count
    return running_ave

def plot_onsager(filename0,filename1):
    onsager = {
        "Lqq":None,
        "Lqm":None,
        "Lmq":None,
        "Lmm":None
    }
    fig,axes = plt.subplots(2,2,figsize=(10,10))
    ylabels = {
               0: ["$L_{qi}(W/m/K)$","$L_{1i}(10^{-6} kg/smK)$"],
               1: ["$L_{1i}(10^{-6} kg/smK)$","$10^{-12} kgs/m^3K$"]
              } 
    ####################################
    # read md0's onsager.txt
    ####################################
    data = np.loadtxt(filename0)[5000:]
    t = np.arange(len(data))*time_step*output_interval*0.001 # ps

    ####################################
    # plot Lqq
    ####################################
    j = data[:,0]
    err = get_ste(j.reshape(100,-1).mean(axis=-1))
    running_j = get_running(j)
    axes[0,0].plot(t,j,c="grey",label="Lqq")
    axes[0,0].plot(t,running_j,c="r",label="running ave")
    axes[0,0].set_xlim(t.min(),t.max())
    axes[0,0].set_xlabel("Running time(ps)")
    axes[0,0].set_ylabel(ylabels[0][0])
    axes[0,0].legend()
    print(f"Lqq: {running_j[-1]}, err: {err}")
    onsager["Lqq"] = j.reshape(100,-1).mean(axis=-1)

    ####################################
    # plot Lqm
    ####################################
    j = data[:,3]
    err = get_ste(j.reshape(100,-1).mean(axis=-1))
    running_j = get_running(j)
    axes[0,1].plot(t,j,c="grey",label="Lqm")
    axes[0,1].plot(t,running_j,c="r",label="running ave")
    axes[0,1].set_xlim(t.min(),t.max())
    axes[0,1].set_xlabel("Running time(ps)")
    axes[0,1].set_ylabel(ylabels[0][1])
    axes[0,1].legend()
    print(f"Lqm: {running_j[-1]}, err: {err}")
    onsager["Lqm"] = j.reshape(100,-1).mean(axis=-1)

    ####################################
    # read md1's onsager.txt
    ####################################
    data = np.loadtxt(filename1)[5000:]
    t = np.arange(len(data))*time_step*output_interval*0.001 # ps

    ####################################
    # plot Lmq
    ####################################
    j = data[:,0]
    err = get_ste(j.reshape(100,-1).mean(axis=-1))
    running_j = get_running(j)
    axes[1,0].plot(t,j,c="grey",label="Lmq")
    axes[1,0].plot(t,running_j,c="r",label="running ave")
    axes[1,0].set_xlim(t.min(),t.max())
    axes[1,0].set_xlabel("Running time(ps)")
    axes[1,0].set_ylabel(ylabels[1][0])
    axes[1,0].legend()
    print(f"Lmq: {running_j[-1]}, err: {err}")
    onsager["Lmq"] = j.reshape(100,-1).mean(axis=-1)

    ####################################
    # plot Lqm
    ####################################
    j = data[:,3]
    err = get_ste(j.reshape(100,-1).mean(axis=-1))
    running_j = get_running(j)
    axes[1,1].plot(t,j,c="grey",label="Lmm")
    axes[1,1].plot(t,running_j,c="r",label="running ave")
    axes[1,1].set_xlim(t.min(),t.max())
    axes[1,1].set_xlabel("Running time(ps)")
    axes[1,1].set_ylabel(ylabels[1][1])
    axes[1,1].legend()
    print(f"Lmm: {running_j[-1]}, err: {err}")
    onsager["Lmm"] = j.reshape(100,-1).mean(axis=-1)

    ####################################
    # save figure and calculate kappa
    ####################################
    plt.savefig("onsager.png")
    kappas = onsager["Lqq"]-onsager["Lqm"]*onsager["Lmq"]/onsager["Lmm"]
    err = get_ste(kappas)
    print(f"kappa: {kappas.mean()}, err: {err}")

time_step = 1
output_interval = 10
filename0 = "../md0/md-0/onsager.out"
filename1 = "../md1/md-0/onsager.out"
plot_onsager(filename0,filename1)
