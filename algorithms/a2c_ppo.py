#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools import frames, losses

# some necesary constants
from algorithms.configs import ALGORITHM_SEED
from sim_env.configs import N_NODES

# and mathematical help
import numpy as np
import tensorflow as tf

# and to make it reproducible
frames.set_tf_seed(ALGORITHM_SEED)

class A2C_PPO_Agent(object):
	"""A2C_PPO_Agent
	"""
	def __init__(self, n):
		super(A2C_PPO_Agent, self).__init__()
		action_possibilities = np.append([N_NODES+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units, n._avail_ram_units)+1 for _ in range(n.max_k)])
		action_possibilities = np.array(action_possibilities, dtype=np.uint8)
		# actual agent - the NN
		self.network = frames.Simple_Frame(action_possibilities)
		# meta-data
		# ...

	def __call__(self, obs, batches=1):
		# wrapp in batches
		if batches == 1:
			obs = tf.expand_dims(obs, 0)
		# call its network
		action_logits_t,_ = self.network(obs)
		# and decipher the action
		action_i = []
		# Since it's multi-discrete, for every discrete set of actions:
		for action_logits_t_k in action_logits_t:
			# Sample next action from the action probability distribution
			action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
			action_i.append(action_i_k.numpy())
		return np.array(action_i)

	def train(self):
		# implement the trainning of the agent
		pass
