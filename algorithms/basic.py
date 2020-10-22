# imports from our utils and modules
from fog import node
from fog import configs
from tools import utils

# external
import numpy as np

class RandomAlgorithm(object):
	"""docstring for RandomAlgorithm"""
	def __str__(self):
		return "Random Algorithm"

	def execute(self,state):
		"""Gives a random action based on the state and limited possible actions
		"""	
		# unpack it
		(w_i, q_i) = state
		# prepare action arrays
		n_o = np.zeros(configs.N_NODES, dtype=np.uint8)
		w_o = np.zeros(configs.N_NODES, dtype=np.uint8)

		for j in range(configs.N_NODES):
			if w_i[j] == 0:
				n_o[j] = 0
				w_o[j] = 0
				continue
			wqj = w_i[j] + q_i[j]
			# check possible acitons
			possible_n_o = [i for i, (w,q) in enumerate(zip(w_i, q_i)) if (i != j and w+q < wqj)]
			if possible_n_o:
				n_o[j] = utils.randomChoice(possible_n_o)
				w_o[j] = w_i[j]
			else:
				n_o[j] = 0
				w_o[j] = 0

		action = np.array((n_o, w_o))
		return action


class LeastQueueAlgorithm(object):
	"""docstring for LeastQueueAlgorithm"""
	def __str__(self):
		return "Least Queue Algorithm"

	def execute(self,state):
		"""Offloads tasks to the node with the minimum queue status
		"""
		# unpack it
		(w_i, q_i) = state
		# prepare action arrays
		n_o = np.zeros(configs.N_NODES, dtype=np.uint8)
		w_o = np.zeros(configs.N_NODES, dtype=np.uint8)

		for j in range(configs.N_NODES):
			if w_i[j] == 0:
				n_o[j] = 0
				w_o[j] = 0
				continue
			wqj = w_i[j] + q_i[j]
			# check possible acitons
			possible_n_o = [i for i, (w,q) in enumerate(zip(w_i, q_i)) if (i != j and w+q < wqj)]
			if possible_n_o:
				m = configs.MAX_QUEUE +1
				for i in possible_n_o:
					if q_i[i] < m:
						m = q_i[i]
						n_o[j] = i
				w_o[j] = w_i[j]
			else:
				n_o[j] = 0
				w_o[j] = 0

		action = np.array((n_o, w_o))
		return action
