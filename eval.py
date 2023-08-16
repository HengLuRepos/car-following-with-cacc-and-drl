from td3 import TwinDelayedDDPG, np2torch
from config import Config
from vehicle import LeadingVehicle, Vehicle
import numpy as np
from ddpg import DDPG
import matplotlib.pyplot as plt
def eval(ddpg):
    ld = LeadingVehicle()
    following_temp = Vehicle(max_v=33.4)
    ld.reset()
    following_temp.reset()
    ep_reward = 0
    done = False
    rel_v = [following_temp.get_state()[1] - following_temp.get_state()[0]]
    pre_v = [following_temp.get_state()[1]]
    rel_d = [following_temp.get_state()[4]]
    steps = 0
    while not done:
        ld_acc = ld.get_acc()
        following_temp.update(ld_acc)
        _, ld_done = ld.step()
        action = ddpg.actor(np2torch(following_temp.get_state())).detach().cpu().numpy()
        next_state, reward, terminated, truncated, _ = following_temp.step(action[0])
        done = terminated or truncated or ld_done
        ld_acc = ld.get_acc()
        following_temp.update(ld_acc)
        ep_reward += reward
        steps += 1
        rel_v.append(following_temp.get_state()[1] - following_temp.get_state()[0])
        pre_v.append(following_temp.get_state()[1])
        rel_d.append(following_temp.get_state()[4])
    print("---------------------------------------")
    print(f"Episodic reward:{ep_reward:.3f} steps:{steps:.3f}")
    print(f"max rel_dist:{max(rel_d):.3f};  min rel_dist:{min(rel_d):.3f} mean rel_dist:{np.mean(rel_d):.3f}")
    print(f"max rel_v:{max(rel_v):.3f};  min rel_v:{min(rel_v):.3f} mean rel_v:{np.mean(rel_v):.3f}")
    print(following_temp.get_state())
    print("---------------------------------------")
    return ep_reward, rel_v, rel_d, steps, pre_v

config = Config()
lead = LeadingVehicle()
following = Vehicle(max_v=config.max_v)
ddpg = DDPG(following, config)
ddpg.load_model("./models/ddpg-cacc.pt")
ep_reward, rel_v, rel_d, steps, pre_v = eval(ddpg)
rel_v = np.array(rel_v)
rel_d = np.array(rel_d)
pre_v = np.array(pre_v)
np.savez("results/ddpg-cacc.npz", v=rel_v,d=rel_d, pre_v = pre_v)