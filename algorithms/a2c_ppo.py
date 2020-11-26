#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame, Actor_Critic_Output_Frame

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_PPO_EPS
from sim_env.configs import N_NODES

# and mathematical help
import numpy as np
import tensorflow as tf


# the class itself
class A2C_PPO_Agent(object):
	"""A2C_PPO_Agent
	"""
	basic = False
	def __init__(self, n, ppo_eps=DEFAULT_PPO_EPS):
		pass


	def __call__(self, obs, batches=1):
		pass

	def model(self, obs):
		pass


