import numpy as np
import matplotlib.pyplot as plt
data = np.load("./results/ddpg-cacc-best.npz")
v = data['v']
d = data['d']
pre_v = data['pre_v']
x = range(len(v))
#plt.figure(1)
#plt.plot(x, v)
#plt.savefig("./figs/td3-cacc-1000-relv.png")
#plt.figure(2)
#plt.plot(x, d)
#plt.savefig("./figs/td3-cacc-1000-reld.png")
plt.plot(x, pre_v, label="pre_v")
plt.plot(x, pre_v - v, label="follow_v")
plt.legend()
plt.savefig("./figs/ddpg-cacc-best-all_v.png")