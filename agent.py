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
