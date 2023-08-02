from ddpg import DDPG, np2torch
from vehicle import LeadingVehicle, Vehicle
import gymnasium as gym
import numpy as np
from config import Config
config = Config(max_acc=3)
NUM_OF_FOLLOWING_CARS = 3


def update_states(follow_states, ld_action, follow_action, followings):
    for i in range(NUM_OF_FOLLOWING_CARS):
        if i == 0:
            follow_states[i] = followings[i].update(ld_action)
        else:
            follow_states[i] = followings[i].update(follow_action[i-1][0])
def explore(ld_action, following_states, followings):
    pre_action = np.array([ld_action])
    actions = []
    for i in range(NUM_OF_FOLLOWING_CARS):
        following_states[i] = followings[i].update(pre_action[0])
        action = ddpg.actor.explore(followings[i].get_state()).detach().cpu().numpy()
        actions.append(action)
        pre_action = action
    return actions

def act(ld_action, following_states, followings):
    pre_action = np.array([ld_action])
    actions = []
    for i in range(NUM_OF_FOLLOWING_CARS):
        following_states[i] = followings[i].update(pre_action[0])
        action = ddpg.actor(np2torch(followings[i].get_state())).detach().cpu().numpy()
        actions.append(action)
        pre_action = action
    return actions

def step(actions, followings):
    next_states, rewards, terminateds, truncateds= [], [], [], []
    for i in range(NUM_OF_FOLLOWING_CARS):
        next_state, reward, terminated, truncated, _ = followings[i].step(actions[i][0])
        next_states.append(next_state)
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)
    return next_states, rewards, terminateds, truncateds
def remember(states, actions, rewards, next_states, dones):
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        ddpg.buffer.remember(state, action, reward, next_state, done)
def eval():
    print("evaluating...")
    ld = LeadingVehicle(iter_steps=100)
    following = [Vehicle(max_acc=config.max_acc, tau=config.tau, max_v=config.max_v) for _ in range(NUM_OF_FOLLOWING_CARS)]
    rel_v = [0.0] * NUM_OF_FOLLOWING_CARS
    dist = [0.0] * NUM_OF_FOLLOWING_CARS
    ep_reward = 0
    for i in range(config.eval_epoch):
        ld_nstate, _ = ld.reset()
        following_state = [env.reset()[0] for env in following]
        rel_v = [state[1] for state in following_state]
        dist = [state[4] for state in following_state]
        done = False
        while not done:
            actions = act(ld.get_acc(), following_states=following_state, followings=following)
            next_states, rewards, terminated, truncated = step(actions=actions, followings=following)
            ld.step()
            done = any(terminated or truncated)
            ep_reward += sum(rewards)
            following_state = next_states
        print("---------------------------------------")
        print(f"Episodic reward:{ep_reward:.3f}")
        print(f"max rel_dist:{max(dist):.3f};  min rel_dist:{min(dist):.3f} mean rel_dist:{np.mean(dist):.3f}")
        print(f"max rel_v:{max(rel_v):.3f};  min rel_v:{min(rel_v):.3f} mean rel_v:{np.mean(rel_v):.3f}")
        print("---------------------------------------")
        ld_nstate, _ = ld.reset()
        following_state = [env.reset()[0] for env in followings]


lead = LeadingVehicle(iter_steps=100)
followings = [Vehicle(max_acc=config.max_acc, tau=config.tau, max_v=config.max_v) for _ in range(NUM_OF_FOLLOWING_CARS)] 
ddpg = DDPG(Vehicle(max_acc=config.max_acc, tau=config.tau, max_v=config.max_v), config)

episode_reward = 0
episode_timesteps = 0
episode_num = 0
ld_state, _ = lead.reset()
following_states = [env.reset()[0] for env in followings]
done = False

for t in range(config.max_timestamp):
    episode_timesteps += 1
    ld_action = lead.get_acc()
    if t < config.start_steps:
        following_actions = [env.action_space.sample() for env in followings]
        update_states(following_states, ld_action, following_actions, followings=followings)
    else:
        following_actions = explore(ld_action, following_states=following_states, followings=followings)
    next_states, rewards, terminated, truncated = step(following_actions, followings=followings)
    ld_state, _ = lead.step()
    does = terminated or truncated
    remember(following_states, following_actions, rewards, next_states, does)
    following_states = next_states
    episode_reward += sum(rewards)
    done = any(does)
    if t >= config.start_steps:
        ddpg.train_iter()
    if done:
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        ld_state, _ = lead.reset()
        following_states = [env.reset()[0] for env in followings]
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if (t + 1) % config.eval_freq == 0:
        eval()

        



