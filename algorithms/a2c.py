#!/usr/bin/env python

# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# and external imports
import tensorflow as tf
import numpy as np

# since we're implementing ppo with deep neural networks
from algorithms.deep_tools.frames import Simple_Frame, Frame_1
from algorithms.deep_tools.common import general_advantage_estimator, normalize_state, map_int_vect_to_int, map_int_to_int_vect

# some necesary constants
from algorithms.configs import ALGORITHM_SEED, DEFAULT_LEARNING_RATE, DEFAULT_ACTION_SPACE
from sim_env.configs import N_NODES, DEFAULT_SLICES, TOTAL_TIME_STEPS


# the class itself
class A2c_Orchestrator(object):
	"""A2c_Orchestrator
	"""
	basic = False
	def __init__(self, env, actor_frame=Frame_1, critic_frame=Frame_1):
		super(A2c_Orchestrator, self).__init__()
		# common critic
		self.critic = critic_frame(1)
		# node actors ~ each actor has two action spaces: for scheduling and for offloading
		self.actors = [actor_frame(map_int_vect_to_int(action_space_n)+1) for action_space_n in env.action_space.nvec]
		self.actors_names = ["_node"+str(n.index) for n in env.nodes]
		self.num_actors = len(env.nodes)

		# meta-data
		self.name = env.case["case"]+"_rd"+str(env.rd_seed)+"_a2c_orchestrator_"+str(self.critic)
		self.action_spaces = env.action_space.nvec
		self.observation_spaces = env.observation_space.nvec

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "a2c"

	def act(self, obs_n):
		# for each agent decide an action
		action = []
		for obs, actor, action_space in zip(obs_n, self.actors, self.action_spaces):
			obs = tf.expand_dims(obs, 0)
			# call its model
			action_logits_t = actor(obs)
			# Since it's multi-discrete, for every discrete set of actions:
			action_i = map_int_to_int_vect(action_space, tf.random.categorical(action_logits_t,1)[0,0].numpy())
			action.append(action_i)
		return np.array(action)
