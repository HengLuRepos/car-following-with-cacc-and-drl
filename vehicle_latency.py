import gymnasium as gym
import numpy as np
import math
from collections import deque
"""
environment for running vehicles
ignore change rate of acc
"""
class Vehicle:
  """
  observation: distance, velocity, acc, relative v(m/s), relative a, relative d(m)
  action: acceleration
  latency: V2V communication latency(fixed, int)
  """
  def __init__(self, max_acc=3, tau=0.8, max_v=33.3, latency=1):
    self.max_acc = max_acc
    self.tau = tau
    self.max_v = max_v
    self.num_steps = 0 
    self.fps = 100
    self.dt = 1/self.fps
    self.steps = 0
    self.buffer = deque(maxlen=latency + 1)
    self.action_space = gym.spaces.Box(low=-self.max_acc, 
                                       high=self.max_acc, 
                                       shape=(1,),
                                       dtype=np.float32)
    ob_low = np.array([
      0,            #velocity
      -self.max_v,  #pre velocity
      -self.max_acc,#acc
      -self.max_acc,#pre acc
      0,            #rel d
    ])

    ob_high = np.array([
      self.max_v,
      self.max_v,
      self.max_acc,
      self.max_acc,
      np.inf
    ])

    self.observation_space = gym.spaces.Box(
      low=ob_low,
      high=ob_high,
      dtype=np.float32
    )

  def reset(self, seed=1):
    self.distance = 0
    self.v = 0
    self.acc = 0
    self.rel_d = 2 #initial distance: 2m
    self.num_steps = 0
    self.steps = 0
    self.pre_v = 0
    self.pre_acc = 0
    self.pre_dist = 0
    state = np.array([self.v, self.pre_v, self.acc, self.pre_acc, self.rel_d])
    info = {'distance': self.distance}
    self.buffer.clear()
    self.buffer.append(self.get_state())
    return state, info

  def update(self, state, dist):
    self.pre_v = state[0]
    self.pre_acc = state[2]
    self.pre_dist = dist
  def get_state(self):
    return np.array([self.v, self.pre_v, self.acc, self.pre_acc, self.rel_d]), self.distance
  
  def step(self, acc):
    #1 step -> 10 ms
    self.acc = acc
    self.buffer.append(self.get_state())
    self.distance += self.acc * self.dt**2 / 2 + self.v * self.dt
    self.pre_dist += self.pre_acc * self.dt ** 2 / 2 + self.pre_v * self.dt
    self.rel_d = self.pre_dist - self.distance + 2
    safe_distance = 2 + self.v * self.tau - self.max_acc * self.tau**2 / 2
    self.v += acc * self.dt
    self.pre_v += self.pre_acc * self.dt
    self.steps += 1
    self.steps %= self.fps
    if self.steps == 0:
      self.num_steps += 1
    def get_reward(rel_v, safe_dist, rel_d):
      return -np.abs(rel_v)/self.max_v - np.abs(rel_d - safe_dist)/safe_dist + self.v/self.max_v + 1
    
    reward = get_reward(self.v - self.pre_v, safe_distance, self.rel_d)
    terminated = self.rel_d <= 0 or self.v > self.max_v or np.abs(self.acc) > self.max_acc or self.v < 0
    truncated = self.num_steps >= 1000
    next_state = np.array([self.v, self.pre_v, self.acc, self.pre_acc, self.rel_d])
    info = {'distance': self.distance}

    return next_state, reward, terminated, truncated, info
  
  def wait(self):
    self.step(0.0)
  
  def try_step(self, acc):
    rel_acc = acc - self.pre_acc
    rel_v = self.v - self.pre_v
    distance = self.distance + self.acc * self.dt ** 2 / 2 + self.v * self.dt
    pre_dist = self.pre_dist + self.pre_acc * self.dt ** 2 / 2 + self.pre_v * self.dt
    rel_d = pre_dist - distance
    safe_distance = 2 + self.v * self.tau - self.max_acc * self.tau**2 / 2
    v = self.v + acc * self.dt
    pre_v = self.pre_v + self.pre_acc
    def get_reward(rel_v, safe_dist, rel_d):
      return -np.abs(rel_v)/self.max_v - np.abs(rel_d - safe_dist)/safe_dist + self.v/self.max_v
    
    reward = get_reward(v - pre_v, safe_distance, rel_d)
    terminated = rel_d <= 0 or v > self.max_v or np.abs(acc) > self.max_acc or v < 0
    truncated = self.num_steps + 1 >= 1000
    next_state = np.array([v, pre_v, acc, self.pre_acc, rel_d])
    info = {'distance': self.distance}

    return reward


#60 80 100 120
class LeadingVehicle(Vehicle):
  def __init__(self, plat_steps=120, max_acc=2.5, tau=0.8, max_v=33.3):
    super().__init__(max_acc=max_acc, tau=tau, max_v=max_v)
    self.plat_steps = plat_steps
    self.acc1 = 16.67 / 10 #10s 0->60km/h
    self.acc1_step = 10
    self.acc2 = 5.55 / 2   #2s 60->80
    self.acc2_step = 2
    self.mid_plat = 1000 - (self.acc1_step * 2 + self.acc2_step * 6 + self.plat_steps * 6)
  def step(self):
    self.acc = self.get_acc()
    self.buffer.append(self.get_state())
    self.steps += 1
    self.steps %= self.fps
    if self.steps == 0:
      self.num_steps += 1
    self.distance += self.acc * self.dt**2 / 2 + self.v * self.dt
    self.v += self.acc * self.dt
    next_state = np.array([round(self.v, 2), round(self.pre_v, 2), self.acc, self.pre_acc, self.rel_d])
    truncated = self.num_steps >= 1000
    return next_state, truncated
  def get_acc(self):
    num_steps = self.num_steps
    if num_steps < self.acc1_step:
      #0->60km/h
      acc = self.acc1
    elif num_steps >= self.plat_steps + self.acc1_step and num_steps < self.plat_steps + self.acc1_step + self.acc2_step:
      #60->80
      acc = self.acc2
    elif num_steps >= 2*self.plat_steps + self.acc1_step + self.acc2_step and num_steps < 2 * self.plat_steps + self.acc1_step + 2 * self.acc2_step:
      #80->100
      acc = self.acc2
    elif num_steps >= 3*self.plat_steps + self.acc1_step + 2*self.acc2_step and num_steps < 3 * self.plat_steps + self.acc1_step + 3 * self.acc2_step:
      #100->120
      acc = self.acc2
    elif num_steps >= 3 * self.plat_steps + self.acc1_step + 3 * self.acc2_step + self.mid_plat and num_steps < 3 * self.plat_steps + self.acc1_step + 4 * self.acc2_step + self.mid_plat:
      #120->100
      acc = -self.acc2
    elif num_steps >= 4 * self.plat_steps + self.acc1_step + 4 * self.acc2_step + self.mid_plat and num_steps < 4 * self.plat_steps + self.acc1_step + 5 * self.acc2_step + self.mid_plat:
      #100->80
      acc = -self.acc2
    elif num_steps >= 5 * self.plat_steps + self.acc1_step + 5 * self.acc2_step + self.mid_plat and num_steps < 5 * self.plat_steps + self.acc1_step + 6 * self.acc2_step + self.mid_plat:
      #80->60
      acc = -self.acc2
    elif num_steps >= 6 * self.plat_steps + self.acc1_step + 6 * self.acc2_step + self.mid_plat and num_steps < 6 * self.plat_steps + 2 * self.acc1_step + 6 * self.acc2_step + self.mid_plat:
      #60->0
      acc = -self.acc1
    else:
      acc = 0
    return acc
  def get_state(self):
    return np.array([self.v, 0, self.acc, 0]), self.distance
car = Vehicle()
ld = LeadingVehicle()
