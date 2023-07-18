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
      0,            #distance
      0,            #velocity
      -self.max_acc,
      0,            #rel d
    ])

    ob_high = np.array([
      np.inf,
      self.max_v,
      self.max_acc,
      np.inf
    ])

    self.observation_space = gym.spaces.Box(
      low=ob_low,
      high=ob_high,
      dtype=np.float32
    )

  def reset(self):
    self.distance = 0
    self.v = 0
    self.acc = 0
    self.rel_d = 2 #initial distance: 2m
    self.num_steps = 0
    state = np.array([self.distance, self.v, self.acc, self.rel_d])
    info = None
    return state, info

  
  def step(self, action):
    #1 step -> 1 sec
    acc, pre_v, pre_acc = action
    """
    pre_v: velocity of preceding vehicle(before accelration)
    pre_acc: acc of preceding vehicle(after changing)
    """
    self.acc = acc
    rel_v = self.v - pre_v
    rel_acc = self.acc - pre_acc
    self.distance += acc / 2 + self.v
    self.rel_d -= rel_v + rel_acc / 2
    safe_distance = 2 + self.max_acc * self.tau**2 / 2 + self.v * self.tau
    self.num_steps += 1
    def get_reward(dist, safe_dist, rel_d):
      return 0
    
    reward = get_reward(self.distance, safe_distance, self.rel_d)
    terminated = self.rel_d <= 0 #collision
    truncated = self.num_steps >= 1000
    next_state = np.array([self.distance, self.v, self.acc, self.rel_d])
    info = None

    return next_state, reward, terminated, truncated, info



class LeadingVehicle(Vehicle):
  def __init__(self, epoch_steps=100):
    super().__init__()
    self.epoch_steps = epoch_steps
    self.acc1 = self.max_acc / 3
    self.acc1_step = math.floor(self.max_v / self.acc1)
    self.acc2 = self.max_acc / 2
    self.acc2_step = math.floor(self.max_v / self.acc2)
  def step(self):
    if self.num_steps % self.epoch_steps < self.acc1_step:
      self.acc = self.acc1
    elif self.epoch_steps - (self.num_steps % self.epoch_steps) <= self.acc2_step:
      self.acc = -self.acc2
    else:
      self.acc = 0
    self.num_steps += 1
    self.distance += self.acc / 2 + self.v
    self.v += self.acc
    next_state = (self.distance, self.v, self.acc)
    truncated = self.num_steps >= 1000
    return next_state, truncated
  def reset(self):
    self.distance = 0
    self.v = 0
    self.acc = 0
    self.num_steps = 0
    state = (self.distance, self.v, self.acc)
    return state
  
ld = LeadingVehicle()
distance, v0, acc0 = ld.reset()
done = False
while not done:
  next_state, truncated = ld.step()
  dist, v, acc = next_state
  done = truncated

