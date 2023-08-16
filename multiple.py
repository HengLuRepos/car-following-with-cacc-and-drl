from ddpg import DDPG, np2torch
from vehicle import LeadingVehicle, Vehicle
import gymnasium as gym
import numpy as np
from config import Config
from td3 import TwinDelayedDDPG
NUM_CAR = 3
def reset(following):
    for env in following:
        env.reset()
def eval():
    ld = LeadingVehicle()
    following_temp = [Vehicle(max_v=33.4) for _ in range(NUM_CAR)]
    ld.reset()
    reset(following_temp)
    ep_reward = 0
    done = False
    rel_v = [following_temp.get_state()[1] - following_temp.get_state()[0]]
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
        rel_d.append(following_temp.get_state()[4])
    print("---------------------------------------")
    print(f"Episodic reward:{ep_reward:.3f} steps:{steps:.3f}")
    print(f"max rel_dist:{max(rel_d):.3f};  min rel_dist:{min(rel_d):.3f} mean rel_dist:{np.mean(rel_d):.3f}")
    print(f"max rel_v:{max(rel_v):.3f};  min rel_v:{min(rel_v):.3f} mean rel_v:{np.mean(rel_v):.3f}")
    print(following_temp.get_state())
    print("---------------------------------------")
    return ep_reward
    

config = Config()
lead = LeadingVehicle()
following = [Vehicle(max_v=33.4) for _ in range(NUM_CAR)]
ddpg = TwinDelayedDDPG(Vehicle(max_v=33.4), config)
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
    ld_action = lead.get_acc()
    following.update(ld_action)
    state = following.get_state()
    if t < config.start_steps:
        following_action = np.abs(following.action_space.sample())
    else:
        following_action = ddpg.actor.explore(np2torch(state)).detach().cpu().numpy()
        current = following.get_state()
        cacc_action = np.array([current[0] + 0.2 * current[4] + 0.9*current[1]])
        if following.try_step(cacc_action[0]) > following.try_step(following_action[0]):
            #following_action = (1.0 - BETA) * cacc_action + BETA * following_action
            following_action = cacc_action
        
    _, ld_done = lead.step()
    next_state, reward, terminated, truncated, _ = following.step(following_action[0])
    next_acc = lead.get_acc()
    next_state[3] = next_acc
    done = terminated or truncated or ld_done
    ddpg.buffer.remember(state, following_action, reward, next_state, done)
    episode_reward += reward

    if t >= config.start_steps:
        for i in range(config.update_freq):
            ddpg.train_iter()
    if done:
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        lead.reset()
        following.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        ddpg.save_model("models/ddpg-cacc-new.pt")
    if (t + 1) % config.eval_freq == 0:
        ep_reward = eval()
        if episodic_reward_eval is None or ep_reward >= episodic_reward_eval:
            episodic_reward_eval = ep_reward
            ddpg.save_model("models/ddpg-cacc-best-new.pt")
        
