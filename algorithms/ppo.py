#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_PPO_EPS, DEFAULT_ACTION_SPACE
from sim_env.configs import N_NODES

# and mathematical help
import numpy as np
import tensorflow as tf


# the class itself
class PPO_Agent(object):
	"""PPO_Agent
	"""
	basic = False
	def __init__(self, n, action_space=DEFAULT_ACTION_SPACE, model_frame=Simple_Frame, ppo_eps=DEFAULT_PPO_EPS):
		super(A2C_Agent, self).__init__()
		# actual agent - the NN
		print(action_space)
		self.model = model_frame(action_space)		
		# meta-data
		self.name = "node_"+str(n)+"_ppo_agent"
		self.action_space = action_space
		self.learning_rate = DEFAULT_LEARNING_RATE
		self.gamma = 0.99
		
		self.ppo_eps = ppo_eps

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "ppo"

	def act(self, obs, batches=1):
		# wrapp in batches
		if batches == 1:
			obs = tf.expand_dims(obs, 0)
		# call its model
		action_logits_t,_ = self.model(obs)
		# Since it's multi-discrete, for every discrete set of actions:
		action_i = []
		for action_logits_t_k in action_logits_t:
			# Sample next action from the action probability distribution
			action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
			action_i.append(action_i_k)
		# return the action for this agent
		return action_i


