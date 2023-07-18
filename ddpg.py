import torch
from torch import nn
from collections import deque
from vehicle import  Vehicle, LeadingVehicle
import random
import numpy as np
def build_mlp(input_dim, output_dim, n_layers, layer_size, activition='relu'):
  net = nn.Sequential(
    nn.Linear(input_dim, layer_size),
    nn.ReLU(),
    *([nn.Linear(layer_size,layer_size), nn.ReLU()]*(n_layers - 1)),
    nn.Linear(layer_size, output_dim),
  )
  if activition is 'relu':
    net.append(nn.ReLU())
  else: 
    net.append(nn.Tanh())
  return net

def np2torch(arr):
  return torch.from_numpy(arr)

class ReplayBuffer:
  def __init__(self, config):
    self.max_len= config.buffer_size
    self.batch_size = config.batch_size
    self.memory = deque(maxlen=self.max_len)
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  
  def sample(self):
    minibatch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for state, action, reward, next_state, done in minibatch:
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      next_states.append(next_state)
      dones.append(done)
    states = np.stack(states)
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    next_states = np.stack(next_states)
    dones = np.stack(dones)
    return states, actions, rewards, next_states, dones
    

class DDPG(nn.Modules):
  def __init__(self, observation_size, action_size, config):
    super().__init__()
    self.observation_size = observation_size
    self.action_size = action_size
    self.config = config
    self.gamma = self.config.gamma
    self.lr = self.config.lr
    self.rho = self.config.rho
    self.q_network = build_mlp(self.observation_size + self.action_size,
                               1, 
                               self.config.n_layers, 
                               self.config.layer_size)
    self.target_q = build_mlp(self.observation_size + self.action_size,
                              1, 
                              self.config.n_layers, 
                              self.config.layer_size)
    self.mu_network = build_mlp(self.observation_size,
                                self.action_size,
                                self.config.n_layers,
                                self.config.layer_size,
                                activition='tanh')
    self.target_mu =  build_mlp(self.observation_size,
                                self.action_size,
                                self.config.n_layers,
                                self.config.layer_size,
                                activition='tanh')
    self.target_q.load_state_dict(self.q_network.state_dict())
    self.target_mu.load_state_dict(self.mu_network.state_dict())

    self.optimizer_q = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
    self.optimizer_mu = torch.optim.Adam(self.mu_network.parameters(), lr=self.lr)
  def save_model(self):
    torch.save(self.state_dict(), './models/ddpg.pt')
  def load_model(self, path='./models/ddpg.pt'):
    self.load_state_dict(torch.load(path))
  
  def grad_q(self, q_values, targets):
    loss = torch.nn.functional.mse_loss(q_values, targets)
    self.optimizer_q.zero_grad()
    loss.backward()
    self.optimizer_q.step()
  
  def grad_mu(self, states):
    states = np2torch(states)
    mus = self.mu_network(states)
    inputs = torch.stack([states, mus], dim=1)
    qs = self.q_network(inputs)
    loss = -qs.mean()
    self.optimizer_mu.zero_grad()
    loss.backward()
    self.optimizer_mu.step()
  
  def update_target(self):
    with torch.no_grad():
      for param1, param2 in zip(self.q_network.parameters(), self.target_q.parameters()):
        param2.copy_(self.rho * param2 + (1.0 - self.rho) * param1)
    
    with torch.no_grad():
      for param1, param2 in zip(self.mu_network.parameters(), self.target_mu.parameters()):
        param2.copy_(self.rho * param2 + (1.0 - self.rho) * param1)


  
  
  
  


    
