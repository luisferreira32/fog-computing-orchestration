# imports from our utils and modules
from fog import node
from fog import configs
from tools import utils
from . import qlearning

class RandomAlgorithm(object):
	"""docstring for RandomAlgorithm"""
	def __init__(self, nodes = None):
		self.nodes = nodes
		self.updatable = False
		self.reward_fun = qlearning.reward

	def __str__(self):
		return "Random Algorithm"

	def setnodes(self, nodes):
		self.nodes = nodes


	def execute(self,state):
		"""Gives a random action based on the state and limited possible actions
		"""	
		# unpack it
		(nL, w, Qsizes) = state

		# check possible acitons
		possible_nO = []
		for i in range(0, len(Qsizes)):
			if Qsizes[i] <= Qsizes[nL.index] and i != nL.index:
				possible_nO.append(i)
		if possible_nO:
			nO_index = utils.randomChoice(possible_nO)
			n0 = self.nodes[nO_index]
			#  unload a random within the possible round(utils.uniformRandom(w))
			w0 = w
		else:
			w0 = 0
			n0 = None

		return [w0, n0]


class LeastQueueAlgorithm(object):
	"""docstring for LeastQueueAlgorithm"""
	def __init__(self, nodes=None):
		self.nodes = nodes
		self.updatable = False
		self.reward_fun = qlearning.reward

	def __str__(self):
		return "Least Queue Algorithm"

	def setnodes(self, nodes):
		self.nodes = nodes
		
	def execute(self,state):
		"""Offloads tasks to the node with the minimum queue status
		"""
		# unpack it
		(nL, w, Qsizes) = state

		# check possible acitons
		possible_nO = []
		possible_nO_q = []
		for i in range(0,len(Qsizes)):
			if Qsizes[i] <= Qsizes[nL.index] and i != nL.index: 
				possible_nO.append(self.nodes[i])
				possible_nO_q.append(Qsizes[i])
		n0 = None
		if possible_nO:
			min_value = min(possible_nO_q)
			indeces = [index for index, value in enumerate(possible_nO_q) if value == min_value]
			i = utils.randomChoice(indeces)
			n0 = possible_nO[i]
		
		w0 = 0
		if n0 is not None:
			w0 = w

		return [w0, n0]

class NearestNodeAlgorithm(object):
	"""docstring for NearestNodeAlgorithm"""
	def __init__(self, nodes=None):
		self.nodes = nodes
		self.updatable = False
		self.reward_fun = qlearning.reward

	def __str__(self):
		return "Nearest Node Algorithm"

	def setnodes(self, nodes):
		self.nodes = nodes
		
	def execute(self,state):
		"""Offloads tasks to the node with the minimum distance to this one, and space on queue
		"""
		# unpack it
		(nL, w, Qsizes) = state

		# check possible actions
		possible_nO = []
		for i in range(0, len(Qsizes)):
			if Qsizes[i] <= nL.qs() and i != nL.index:
				possible_nO.append(i)

		# send to nearest with a lesser queue
		e0 = 9999999
		n0 = None
		if possible_nO: nO_index = possible_nO[0]
		for n,e in nL.comtime.items():
			if n.index in possible_nO and e < e0:
				n0 = n
				e0 = e
		
		w0 = 0
		if n0 is not None:
			w0 = w

		return [w0, n0]