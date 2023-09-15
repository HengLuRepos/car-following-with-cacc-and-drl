from td3 import TwinDelayedDDPG, np2torch
from config import Config
from vehicle_latency import LeadingVehicle, Vehicle
import numpy as np
from ddpg import DDPG
import matplotlib.pyplot as plt
NUM_CAR = 5
def reset(following):
    for env in following:
        env.reset()
def inference(ld: LeadingVehicle, following: list[Vehicle]):
    done = False
    reward = 0.0
    pre_acc = ld.buffer[0]
    rel_v = [[] for _ in range(NUM_CAR)]
    pre_v = [[] for _ in range(NUM_CAR)]
    rel_d = [[] for _ in range(NUM_CAR)] #distance computed
    dists = [[] for _ in range(NUM_CAR)] #actual distance
    for i in range(NUM_CAR):
        following[i].update(*pre_acc)
        pre_acc = following[i].buffer[0]
        action = ddpg.actor(np2torch(following[i].get_state()[0])).detach().cpu().numpy()
        _, r, terminated, truncated, _ = following[i].step(action[0])
        done = done or terminated or truncated
        reward += r
        curr_state, curr_dist = following[i].get_state()
        rel_v[i].append(curr_state[1] - curr_state[0])
        pre_v[i].append(curr_state[1])
        rel_d[i].append(curr_state[4])
        dists[i].append(curr_dist)
    return reward/NUM_CAR, done, rel_v, pre_v, rel_d, dists
def eval(ddpg):
    ld = LeadingVehicle()
    following_temp = [Vehicle(max_v=33.4) for _ in range(NUM_CAR)]
    ld.reset()
    reset(following_temp)
    ep_reward = 0
    done = False
    ld_state, ld_dist = ld.get_state()
    ld_v = [ld_state[0]]
    ld_d = [ld_dist]
    rel_v = [[car.get_state()[0][1] - car.get_state()[0][0]] for car in following_temp]
    pre_v = [[car.get_state()[0][1]] for car in following_temp]
    rel_d = [[car.get_state()[0][4]] for car in following_temp]
    dists = [[0.0] for car in following_temp]
    steps = 0
    while not done:
        reward, done, rel_v_temp, pre_v_temp, rel_d_temp, dist_temp = inference(ld, following_temp)
        _, ld_done = ld.step()
        ld_state, ld_dist = ld.get_state()
        done = done or ld_done
        ep_reward += reward
        steps += 1
        rel_v = [a + b for a, b in zip(rel_v, rel_v_temp)]
        pre_v = [a + b for a, b in zip(pre_v, pre_v_temp)]
        rel_d = [a + b for a, b in zip(rel_d, rel_d_temp)]
        dists = [a + b for a, b in zip(dists, dist_temp)]
        ld_v.append(ld_state[0])
        ld_d.append(ld_dist)
        if steps % 1000 == 0:
            print(steps)
    print("---------------------------------------")
    print(f"Episodic reward:{ep_reward:.3f} steps:{steps:.3f}")
    #print(f"max rel_dist:{max(rel_d):.3f};  min rel_dist:{min(rel_d):.3f} mean rel_dist:{np.mean(rel_d):.3f}")
    #print(f"max rel_v:{max(rel_v):.3f};  min rel_v:{min(rel_v):.3f} mean rel_v:{np.mean(rel_v):.3f}")
    #print(following_temp.get_state())
    print("---------------------------------------")
    return ep_reward, rel_v, rel_d, steps, pre_v, ld_d, ld_v, dists

config = Config()
lead = LeadingVehicle()
following = [Vehicle(max_v=33.4) for _ in range(NUM_CAR)]
ddpg = TwinDelayedDDPG(Vehicle(max_v=33.4), config)
ddpg.load_model("./models/td3-cacc-5-best-latency.pt")
#eval(ddpg)
ep_reward, rel_v, rel_d, steps, pre_v, ld_d, ld_v, dists = eval(ddpg)
rel_v = np.array(rel_v)
rel_d = np.array(rel_d)
pre_v = np.array(pre_v)
ld_d = np.array(ld_d)
ld_v = np.array(ld_v)
dists = np.array(dists)
np.savez(f"results/td3-cacc-{NUM_CAR}.npz", rel_v=rel_v,rel_d=rel_d, pre_v=pre_v, ld_d=ld_d, ld_v=ld_v, dist=dists)