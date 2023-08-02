from ddpg import DDPG
from vehicle import LeadingVehicle, Vehicle
import gymnasium as gym
import numpy as np
from config import Config
config = Config(max_acc=3)
NUM_OF_FOLLOWING_CARS = 3
lead = LeadingVehicle(iter_steps=100)
followings = [Vehicle(max_acc=config.max_acc, tau=config.tau, max_v=config.max_v)] * NUM_OF_FOLLOWING_CARS
ddpg = DDPG(Vehicle(max_acc=config.max_acc, tau=config.tau, max_v=config.max_v), config)

episode_reward = 0
episode_timesteps = 0
episode_num = 0
ld_state, _ = lead.reset()
following_states = [env.reset()[0] for env in followings]
done = False

def update_states(follow_states, ld_action, follow_action):
    for i in range(NUM_OF_FOLLOWING_CARS):
        if i == 0:
            follow_states[i] = followings[i].update(ld_action)
        else:
            follow_states[i] = followings[i].update(follow_action[i-1])
def explore(ld_action):
    pre_action = np.array([ld_action])
    actions = []
    for i in range(NUM_OF_FOLLOWING_CARS):
        following_states[i] = followings[i].update(pre_action[0])
        action = ddpg.actor.explore(followings[i].get_state()).detach().cpu().numpy()
        actions.append(action)
        pre_action = action
    return actions
def step(actions):
    next_states, rewards, terminateds, truncateds= [], [], [], []
    for i in range(NUM_OF_FOLLOWING_CARS):
        next_state, reward, terminated, truncated, _ = followings[i].step(actions[i])
        next_states.append(next_state)
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)


for t in range(config.max_timestamp):
    episode_timesteps += 1
    ld_action = lead.get_acc()
    if t < config.start_steps:
        following_actions = [env.action_space.sample() for env in followings]
        update_states(following_states, ld_action, following_actions)
    else:
        following_actions = explore(ld_action)
    next_states, rewards, terminated, truncated= [], [], [], []

        



