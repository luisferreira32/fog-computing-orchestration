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
		super(A2C_PPO_Agent, self).__init__()
		action_possibilities = np.append([N_NODES+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units, n._avail_ram_units)+1 for _ in range(n.max_k)])
		action_possibilities = np.array(action_possibilities, dtype=np.uint8)
		# actual agent - the NN
		self.input_model = Simple_Frame()
		self.output_model = Actor_Critic_Output_Frame(action_possibilities)
		
		# meta-data
		self.learning_rate = DEFAULT_LEARNING_RATE
		self.ppo_eps = ppo_eps


	def __call__(self, obs, batches=1):
		# wrapp in batches
		if batches == 1:
			obs = tf.expand_dims(obs, 0)
		# call its model
		action_logits_t,_ = self.model(obs)
		# and decipher the action
		action_i = []
		# Since it's multi-discrete, for every discrete set of actions:
		for action_logits_t_k in action_logits_t:
			# Sample next action from the action probability distribution
			action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
			action_i.append(action_i_k.numpy())
		# return the action for this agent
		return np.array(action_i)

	def model(self, obs):
		return self.output_model(self.input_model(obs))


