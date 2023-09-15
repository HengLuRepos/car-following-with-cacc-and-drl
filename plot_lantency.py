import numpy as np
import matplotlib.pyplot as plt
data = np.load("./results/td3-cacc-5.npz")
rel_v = data['rel_v']
rel_d = data['rel_d']
pre_v = data['pre_v']
ld_d = data["ld_d"]
ld_v = data['ld_v']
dist = data['dist']
real_d0 = ld_d - dist[0] + 2
real_d1 = dist[0] - dist[1] + 2
real_d2 = dist[1] - dist[2] + 2
real_d3 = dist[2] - dist[3] + 2
real_d4 = dist[3] - dist[4] + 2
x = np.linspace(start=0.0, stop=1000.0, num=len(ld_d))
#plt.figure(1)
#plt.plot(x, v)
#plt.savefig("./figs/td3-cacc-1000-relv.png")
#plt.figure(2)
#plt.plot(x, d)
#plt.savefig("./figs/td3-cacc-1000-reld.png")
plt.plot(x, real_d0, label="rel_d-1")
plt.plot(x, real_d1, '--',label="rel_d-2")
plt.plot(x, real_d2, ':',label="rel_d-3")
plt.plot(x, real_d3, '-.',label="rel_d-4")
plt.plot(x, real_d4, label="rel_d-5")
plt.xlabel("time(s)")
plt.ylabel("distance(m)")
#plt.plot(x, pre_v - v, label="follow_v")
plt.legend()
plt.savefig("./figs/cacc-cacc-5-best-rel_d-latency.png")

plt.figure()
stack_d1 = real_d0 + real_d1
stack_d2 = stack_d1 + real_d2
stack_d3 = stack_d2 + real_d3
stack_d4 = stack_d3 + real_d4
plt.plot(x, real_d0, label="rel_d-1_0")
plt.plot(x, stack_d1, '--',label="rel_d-2_0")
plt.plot(x, stack_d2, ':',label="rel_d-3_0")
plt.plot(x, stack_d3, '-.',label="rel_d-4_0")
plt.plot(x, stack_d4, label="rel_d-5_0")
plt.xlabel("time(s)")
plt.ylabel("distance(m)")
plt.legend()
plt.savefig("./figs/cacc-cacc-5-best-rel_d-latency-stack.png")