#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# and external imports
import tensorflow as tf

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame
from algorithms.deep_tools.common import general_advantage_estimator, normalize_state

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_ACTION_SPACE
from sim_env.configs import N_NODES, DEFAULT_SLICES, TOTAL_TIME_STEPS


# the class itself
class A2C_Agent(object):
	"""A2C_Agent
	"""
	basic = False
	def __init__(self, n, env, model_frame=Simple_Frame):
		super(A2C_Agent, self).__init__()
		# actual agent - the NN
		self.model = model_frame(env.action_space.nvec[n-1])
		# meta-data
		self.name = env.case["case"]+"_rd"+str(env.rd_seed)+"_node_"+str(n)+"_a2c_agent_"+str(self.model)
		self.action_space = tf.constant(env.action_space.nvec[n-1], dtype=tf.uint8)
		self.observation_space_max = tf.constant(env.observation_space.nvec[n-1], dtype=tf.uint8)

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "a2c"

	def act(self, obs, batches=1):
		# wrapp in batches
		if batches == 1:
			obs = normalize_state(obs, self.observation_space_max)
			obs = tf.expand_dims(obs, 0)
		# call its model
		model_output = self.model(obs)
		action_logits_t = model_output[:-1]
		# Since it's multi-discrete, for every discrete set of actions:
		action_i = []
		for action_logits_t_k in action_logits_t:
			# Sample next action from the action probability distribution
			action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
			action_i.append(action_i_k)
		# return the action for this agent
		return action_i
