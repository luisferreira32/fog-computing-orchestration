#!/usr/bin/env python


class Dqn_Orchestrator(object):
	"""docstring for Dqn_Orchestrator"""
	basic = False
	def __init__(self, env):
		super(Dqn_Orchestrator, self).__init__()

		# N DQN for each node
		# N DQN target for each node too

		# meta-data
		self.epsilon = 0.9#INITIAL EPSILON

	def __str__(self):
		return self.name

	@staticmethod
	def short_str():
		return "dqn"

	def act(self, obs_n):
		# given an observation and the policy (e-greedy) return the action
		pass

	def save_models(self, saved_models_path):
		pass

	def load_models(self, saved_models_path):
		pass

	def train(self):
		# init a replay buffer with fixed size
		# run for 10^4 iterations from random sampling to the replay buffer
		
		# update epsilon
		# now for each iteration run (with policy on DQNs) and save state-action-reward-state'
		# sample random mini-batch and perform gradients to update network
		# every C steps replace target network with trained network

		# update epsilon updating parameters every R_E

		pass
		