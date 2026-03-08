import matplotlib.pyplot as plt
import sys
import numpy as np

def get_ste(y):
    std = np.std(y)
    n = len(y)
    ste = std/(n**0.5)
    return ste

filename = sys.argv[1]
time_step = 1 #fs
data = np.loadtxt(filename)

n = int(0.5*(len(data[0])-1))
offset = n+1
t = data[:,0]
ac = data[:,1:offset]
rtc = data[:,offset:]
plt.plot(t,rtc,alpha=0.5,c="grey")
plt.plot(t,rtc.mean(axis=-1),c="r",linewidth=2)
plt.xlim(t.min(),t.max())
plt.xlabel("Correlation time(ps)")
plt.savefig("correlate.png")

value_mean = rtc.mean(axis=-1)[-1]
error = get_ste(rtc[-1])
print(f"{filename}: {value_mean}±{error}")
