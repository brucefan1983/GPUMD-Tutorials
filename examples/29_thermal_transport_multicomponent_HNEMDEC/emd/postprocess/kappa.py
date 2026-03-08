import numpy as np

dict_map = {
    "Lqq":"cor_tt.out",
    "Lqm":"cor_tm.out",
    "Lmq":"cor_mt.out",
    "Lmm":"cor_mm.out"
}
onsager = {
    "Lqq":None,
    "Lqm":None,
    "Lmq":None,
    "Lmm":None
}
for coe,filename in dict_map.items():
    data = np.loadtxt(filename)

    n = int(0.5*(len(data[0])-1))
    offset = n+1
    onsager[coe] = data[-1,offset:]
kappas = onsager["Lqq"]-onsager["Lqm"]*onsager["Lmq"]/onsager["Lmm"]

kappa = kappas.mean()
err = kappas.std()/(len(kappas)**0.5)
print(f"thermal conductivity: {kappa} (W/m/K)")
print(f"error: {err}")
