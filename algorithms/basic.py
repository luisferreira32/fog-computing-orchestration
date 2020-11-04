import numpy as np

from sim_env.envrionment import split_observation_by_node
from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE, RAM_UNIT

class Nearest_Round_Robin(object):
	"""Nearest_Round_Robin this algorithm allows every slice that
	has tasks in the queue to take turns in processing on a
	shared resource in a periodically repeated order. Offloading when
	the queue is 0.8 full to the nearest node
	"""
	def __init__(self, env):
		# to keep track of what slice is processing
		self.process = [ 0 for n in range(N_NODES)]
		self.nodes = env.nodes

	def __str__(self):
		return "Round Robin"

	def predict(self, obs):
		# action struct: [f_00, ..., f_0k, w_00, ..., w_0k, ..., f_i0, ..., f_ik, w_i0, ..., w_ik]
		# default no offloads and no processing
		action = []
		for n in range(N_NODES):
			action  = np.append(action, np.append([n for _ in range(DEFAULT_SLICES)], [0 for _ in range(DEFAULT_SLICES)]))
		action = np.array(action, dtype=np.uint8)

		# for every node make an decision
		obs_by_nodes = split_observation_by_node(obs)
		for i in range(N_NODES):
			[a_k, b_k, be_k, rc_k, rm_k] = np.split(obs_by_nodes[i], [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
			
			# to process based on availabe memory and RR priority
			while rm_k >= self.nodes[i]._task_type_on_slices[self.process[i]][2]/RAM_UNIT and not np.all(b_k == be_k):
				if b_k[self.process[i]] > be_k[self.process[i]]:
					# set the w_ik to process +1
					action[i*DEFAULT_SLICES*2+DEFAULT_SLICES+self.process[i]] += 1
					# and take the resources on the available obs
					rm_k -= int(self.nodes[i]._task_type_on_slices[self.process[i]][2]/RAM_UNIT)
					be_k[self.process[i]] += 1

				self.process[i] += 1
				if self.process[i] == DEFAULT_SLICES:
					self.process[i] = 0

			# offload to the Nearest Node if buffer bigger than 0.8
			for k,b in enumerate(b_k):
				if b > 0.8*MAX_QUEUE:
					# set the f_ik to the nearest node
					action[i*DEFAULT_SLICES*2+k] = self.nodes[i]._communication_rates.index(max(self.nodes[i]._communication_rates))

		# and return the action
		return action
