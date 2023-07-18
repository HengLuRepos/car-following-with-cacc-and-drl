import torch
from torch import nn
from collections import deque
from vehicle import  Vehicle, LeadingVehicle

def build_mlp(input_dim, n_layers, layer_size):
  return nn.Sequential(
    nn.Linear(input_dim, layer_size),
    nn.ReLU(),
    *([nn.Linear(layer_size,layer_size), nn.ReLU()]*(n_layers - 1)),
    nn.Linear(layer_size, 1),
    nn.Tanh()
  )

class DQN:
  def __init__(self, ob_size, config):
    self.observation_size = ob_size
    self.config = config
    self.gamma = self.config.gamma
    self.epsilon = self.config.epsilon
    self.lr = self.config.lr
    self.memory = deque(maxlen=self.config.buffer_size)
    self.q_network = build_mlp(self.observation_size + 1, self.config.n_layers, self.config.layer_size)
    self.target_q = build_mlp(self.observation_size + 1, self.config.n_layers, self.config.layer_size)
    self.target_q.load_state_dict(self.target_q.state_dict())

  def update_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  
  
