import gymnasium as gym
import numpy as np
import math
"""
environment for running vehicles
ignore change rate of acc
"""
class Vehicle:
  """
  observation: distance, velocity, acc, relative v(m/s), relative a, relative d(m)
  action: acceleration
  """
  def __init__(self, max_acc=3, tau=0.8, max_v=30):
    self.max_acc = max_acc
    self.tau = tau
    self.max_v = max_v
    self.num_steps = 0 
    self.action_space = gym.spaces.Box(low=-self.max_acc, 
                                       high=self.max_acc, 
                                       shape=(1,),
                                       dtype=np.float32)
    ob_low = np.array([
      0,            #velocity
      -self.max_v,  #rel velocity
      -self.max_acc,#acc
      -self.max_acc,#rel acc
      0,            #rel d
    ])

    ob_high = np.array([
      self.max_v,
      self.max_v,
      self.max_acc,
      2 * self.max_acc,
      np.inf
    ])

    self.observation_space = gym.spaces.Box(
      low=ob_low,
      high=ob_high,
      dtype=np.float32
    )

  def reset(self, seed):
    self.distance = 0
    self.v = 0
    self.acc = 0
    self.rel_d = 2 #initial distance: 2m
    self.num_steps = 0
    self.rel_v = 0
    self.rel_acc = 0
    state = np.array([self.v, self.rel_v, self.acc, self.rel_acc, self.rel_d])
    info = {'distance': self.distance}
    return state, info

  def update(self, pre_acc):
    self.rel_acc = self.acc - pre_acc
    return np.array([self.v, self.rel_v, self.acc, self.rel_acc, self.rel_d])
  def get_state(self):
    return np.array([self.v, self.rel_v, self.acc, self.rel_acc, self.rel_d])
  
  def step(self, acc):
    #1 step -> 1 sec
    self.acc = acc
    self.distance += self.acc / 2 + self.v
    self.rel_d -= self.rel_v + self.rel_acc / 2
    safe_distance = 2 + self.v * self.tau - self.max_acc * self.tau**2 / 2
    self.v += acc
    self.rel_v += self.rel_acc
    self.num_steps += 1
    def get_reward(rel_v, safe_dist, rel_d):
      return -np.abs(rel_v) - np.abs(rel_d - safe_dist)
    
    reward = get_reward(self.rel_v, safe_distance, self.rel_d)
    terminated = self.rel_d <= 0 #collision
    truncated = self.num_steps >= 1000
    next_state = np.array([self.v, self.rel_v, self.acc, self.rel_acc, self.rel_d])
    info = {'distance': self.distance}

    return next_state, reward, terminated, truncated, info



class LeadingVehicle(Vehicle):
  def __init__(self, iter_steps=100, max_acc=3, tau=0.8, max_v=30):
    super().__init__(max_acc=max_acc, tau=tau, max_v=max_v)
    self.iter_steps = iter_steps
    self.acc1 = self.max_acc / 3
    self.acc1_step = math.floor(self.max_v / self.acc1)
    self.acc2 = self.max_acc / 2
    self.acc2_step = math.floor(self.max_v / self.acc2)
  def step(self):
    self.acc = self.get_acc()
    self.num_steps += 1
    self.distance += self.acc / 2 + self.v
    self.v += self.acc
    next_state = np.array([self.v, self.rel_v, self.acc, self.rel_acc, self.rel_d])
    truncated = self.num_steps >= 1000
    return next_state, truncated
  def get_acc(self):
    num_steps = self.num_steps
    if num_steps % self.iter_steps < self.acc1_step:
      acc = self.acc1
    elif self.iter_steps - (num_steps % self.iter_steps) <= self.acc2_step:
      acc = -self.acc2
    else:
      acc = 0
    return acc

