from ddpg import DDPG, np2torch
from vehicle_latency import LeadingVehicle, Vehicle
import gymnasium as gym
import numpy as np
from config import Config
from td3 import TwinDelayedDDPG
NUM_CAR = 5
def reset(following):
    for env in following:
        env.reset()
def inference(ld: LeadingVehicle, following: list[Vehicle]):
    done = False
    reward = 0.0
    pre = ld.buffer[0]
    for i in range(NUM_CAR):
        following[i].update(*pre)
        pre = following[i].buffer[0]
        action = ddpg.actor(np2torch(following[i].get_state()[0])).detach().cpu().numpy()
        #action = ddpg[i].actor(np2torch(following[i].get_state())).detach().cpu().numpy()
        _, r, terminated, truncated, _ = following[i].step(action[0])
        done = done or terminated or truncated
        reward += r
    return reward/NUM_CAR, done

def eval():
    ld = LeadingVehicle()
    following_temp = [Vehicle(max_v=33.4) for _ in range(NUM_CAR)]
    ld.reset()
    reset(following_temp)
    ep_reward = 0
    done = False
    #rel_v = [following_temp.get_state()[1] - following_temp.get_state()[0]]
    #rel_d = [following_temp.get_state()[4]]
    steps = 0
    while not done:
        reward, done = inference(ld, following_temp)
        _, ld_done = ld.step()
        ep_reward += reward
        steps += 1
        #rel_v.append(following_temp.get_state()[1] - following_temp.get_state()[0])
        #rel_d.append(following_temp.get_state()[4])
    print("---------------------------------------")
    print(f"Episodic reward:{ep_reward:.3f} steps:{steps:.3f}")
    #print(f"max rel_dist:{max(rel_d):.3f};  min rel_dist:{min(rel_d):.3f} mean rel_dist:{np.mean(rel_d):.3f}")
    #print(f"max rel_v:{max(rel_v):.3f};  min rel_v:{min(rel_v):.3f} mean rel_v:{np.mean(rel_v):.3f}")
    #print(following_temp.get_state())
    print("---------------------------------------")
    return ep_reward
    
def act(ld: LeadingVehicle, following: list[Vehicle], start=False):
    pre = ld.buffer[0]
    reward = 0.0
    done = False
    for i in range(NUM_CAR):
        following[i].update(*pre)
        pre_dist = pre[1]
        pre_acc = pre[0][2]
        pre = following[i].buffer[0]
        state = following[i].get_state()[0]
        if pre_dist == 0.0 and pre_acc == 0.0:
            action = np.array([0.0])
        elif start is False:
            action = ddpg.actor.explore(np2torch(state)).detach().cpu().numpy()
            #action = ddpg[i].actor.explore(np2torch(state)).detach().cpu().numpy()
            cacc_action = np.array([state[0] + 0.2*state[4] + 0.9*state[1]])
            if following[i].try_step(cacc_action[0]) > following[i].try_step(action[0]):
                #action = (1.0 - BETA) * cacc_action + BETA * action
                action = cacc_action
            else:
                pass
                #action = (1.0 - BETA) * action + BETA * cacc_action
        else:
            action = np.abs(following[i].action_space.sample())
        next_state, r, terminated, truncated, _ = following[i].step(action[0])
        donei = terminated or truncated
        done = done or donei
        reward += r
        ddpg.buffer.remember(state, action, r, next_state, donei)
        #ddpg[i].buffer.remember(state, action, r, next_state, donei)
    return reward/NUM_CAR, done


config = Config()
lead = LeadingVehicle()
following = [Vehicle(max_v=33.4) for _ in range(NUM_CAR)]
ddpg = TwinDelayedDDPG(Vehicle(max_v=33.4), config)
ddpg.load_model("./models/td3-cacc-2-new.pt")
#ddpg = [TwinDelayedDDPG(Vehicle(max_v=33.4), config) for _ in range(NUM_CAR)]
episode_reward = 0
episode_timesteps = 0
episode_num = 0
ld_state, _ = lead.reset()
reset(following)
BETA = 0.5
cacc = False
episodic_reward_eval = None
for t in range(config.max_timestamp):
    episode_timesteps += 1
    _, ld_done = lead.step()
    if t < config.start_steps:
        reward, done = act(lead, following, start=True)
    else:
        reward, done = act(lead, following)
        
    done = done or ld_done
    episode_reward += reward

    if t >= config.start_steps:
        for i in range(config.update_freq):
            #for agent in ddpg:
            #    agent.train_iter()
            ddpg.train_iter()
    if done:
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        lead.reset()
        reset(following)
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        ddpg.save_model(f"models/td3-cacc-{NUM_CAR}-latency.pt")
        #for i in range(NUM_CAR):
        #    ddpg[i].save_model(f"models/td3-cacc-{NUM_CAR}-part-{i+1}.pt")
    if (t + 1) % config.eval_freq == 0:
        ep_reward = eval()
        if episodic_reward_eval is None or ep_reward >= episodic_reward_eval:
            episodic_reward_eval = ep_reward
            ddpg.save_model(f"models/td3-cacc-{NUM_CAR}-best-latency.pt")
            #for i in range(NUM_CAR):
            #    ddpg[i].save_model(f"models/td3-cacc-{NUM_CAR}-best-part-{i+1}.pt")
        
