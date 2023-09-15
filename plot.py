import numpy as np
import matplotlib.pyplot as plt
data = np.load("./results/td3-cacc-5-original.npz")
v = data['v']
d = data['d']
pre_v = data['pre_v']
x = range(len(v[0]))
d0 = d[0]
d1 = d0 + d[1]
d2 = d1 + d[2]
d3 = d2 + d[3]
d4 = d3 + d[4]
#plt.figure(1)
#plt.plot(x, v)
#plt.savefig("./figs/td3-cacc-1000-relv.png")
#plt.figure(2)
#plt.plot(x, d)
#plt.savefig("./figs/td3-cacc-1000-reld.png")
plt.plot(x, d0, label="rel_d-1_0")
plt.plot(x, d1, '--',label="rel_d-2_0")
plt.plot(x, d2, ':',label="rel_d-3_0")
plt.plot(x, d3, '-.',label="rel_d-4_0")
plt.plot(x, d4, label="rel_d-5_0")
plt.xlabel("time(s)")
plt.ylabel("distance(m)")
#plt.plot(x, pre_v - v, label="follow_v")
plt.legend()
plt.savefig("./figs/cacc-cacc-5-best-rel_d-original.png")