#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# and external imports
import tensorflow as tf
import numpy as np

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame
from algorithms.deep_tools.common import general_advantage_estimator, normalize_state

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_ACTION_SPACE
from sim_env.configs import N_NODES, DEFAULT_SLICES, TOTAL_TIME_STEPS


# the class itself
class A2c_Orchestrator(object):
	"""A2c_Orchestrator
	"""
	basic = False
	def __init__(self, env, actor_frame=Simple_Frame, critic_frame=Simple_Frame):
		super(A2c_Orchestrator, self).__init__()
		# common critic
		self.critic = critic_frame([1])
		# node actors
		self.actors = [actor_frame(action_space_n) for action_space_n in env.action_space.nvec]
		self.actors_names = ["_node"+str(n.index) for n in env.nodes]
		self.num_actors = len(env.nodes)

		# meta-data
		self.name = env.case["case"]+"_rd"+str(env.rd_seed)+"_a2c_orchestrator_"+str(self.critic)
		self.action_space = tf.constant(env.action_space.nvec, dtype=tf.uint8)
		self.observation_space_max = tf.constant(env.observation_space.nvec, dtype=tf.uint8)

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "a2c"

	def act(self, obs_n):
		#obs_n = normalize_state(obs_n, self.observation_space_max)
		# for each agent decide an action
		action = []
		for obs, actor in zip(obs_n, self.actors):
			obs = tf.expand_dims(obs, 0)
			# call its model
			action_logits_t = actor(obs)
			# Since it's multi-discrete, for every discrete set of actions:
			action_i = []
			for action_logits_t_k in action_logits_t:
				# Sample next action from the action probability distribution
				action_i_k = tf.random.categorical(action_logits_t_k,1)[0,0]
				action_i.append(action_i_k.numpy())
			# return the action for this agent
			action.append(action_i)
		return np.array(action)
