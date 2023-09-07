import numpy as np
import matplotlib.pyplot as plt
data = np.load("./results/td3-cacc-5.npz")
v = data['v']
d = data['d']
pre_v = data['pre_v']
x = range(len(v[0]))
#plt.figure(1)
#plt.plot(x, v)
#plt.savefig("./figs/td3-cacc-1000-relv.png")
#plt.figure(2)
#plt.plot(x, d)
#plt.savefig("./figs/td3-cacc-1000-reld.png")
plt.plot(x, v[0], label="rel_v-1")
plt.plot(x, v[1], label="rel_v-2")
plt.plot(x, v[2], label="rel_v-3")
plt.plot(x, v[3], label="rel_v-4")
plt.plot(x, v[4], label="rel_v-5")
#plt.plot(x, pre_v - v, label="follow_v")
plt.legend()
plt.savefig("./figs/cacc-cacc-5-best-rel_v.png")