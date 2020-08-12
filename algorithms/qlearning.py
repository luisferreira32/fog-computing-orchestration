# external imports
import math
import random

# our module imports
from fog import configs

class Qlearning(object):
	"""
	The object of Qlearning, containing all methods and attributes necessary to train, and run a q learning algorithm
	
	...

	Attributes
	----------
	qtable : dict = {state: actions_dict}
		a dictionary that for each state gives a dict of actions

	Methods
	-------
	addstate()
		adds a new state if it wasn't here before, with all possible actions zeroed
	execute()
		chooses a set of actions for this timestep
	update()
		updates the q values from the table
	"""

	def __init__(self):
		# init with an empty dict table
		self.qtable = {}

	def addstate(self, state=None, nodes=None, origin=None):
		""" Adds a new state if it wasn't here before, with all possible actions zeroed

		Parameters
		----------
		nodes=None
			are all the nodes and their current states in the system
		origin=None
			is the node that's offloading
		"""
		if nodes is None or origin is None:
			return -1

		# if the new key is not in the qtable
		if state not in self.qtable:
			# create the dict of q zeroed actions
			actions = {}
			for w0 in range(0,configs.MAX_INFLUX+1):
				for n in nodes:
					if n != origin:
						actions[w0,n.name] = 0
			self.qtable[state] = actions
		else:
			return 1

		return 0

	def execute(self, nodes=None, origin=None, epsilon=None):
		""" Chooses a set of actions to do on the current state

		Parameters
		----------
		nodes=None
			are all the nodes and their current states in the system
		origin=None
			is the node that's offloading

		Returns
		-------
		action
			characterized by origin, destination, w0 to be offloaded. or no action if failed
		"""
		random.seed(17)

		action = []
		if epsilon is None or nodes is None or origin is None:
			return action

		auxq = []
		for n in nodes:
			auxq.append(n.qs())
		state = tuple([origin.name, origin.influx, tuple(auxq)])
		addstate(state, nodes, origin)

		x = random.random()
		# explore
		if x < epsilon:
