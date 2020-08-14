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
	alpha=0.5
		learning rate of the Q table
	discount_factor=0.5
		discount factor of the Q table
	r_utility, x_delay, x_overload
		are scalar constants for the reward calculation

	Methods
	-------
	addstate()
		adds a new state if it wasn't here before, with all possible actions zeroed
	execute()
		chooses a set of actions for this timestep
	reward()
		calculates a reward based on an action and the current state
	update()
		updates the q values from the table
	"""

	def __init__(self, a=0.5, df=0.5):
		# init with an empty dict table
		self.qtable = {}
		self.alpha = a
		self.discount_factor = df
		self.r_utility = 10
		self.x_delay = 1
		self.x_overload = 150

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
						actions[w0,n] = 0
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
		action = []
		if epsilon is None or nodes is None or origin is None:
			return action

		auxq = []
		for n in nodes:
			auxq.append(n.qs())
		state = tuple([origin.name, origin.influx, tuple(auxq)])
		self.addstate(state, nodes, origin)

		x = random.random()
		# explore
		if x < epsilon:
			(w0, dest) = random.choice(list(self.qtable[state]))
			action = [origin, dest, w0]
		# exploit
		else:
			(w0, dest) = max(self.qtable[state], key=self.qtable[state].get)
			action = [origin, dest, w0]

		action_reward = reward(nodes, origin, action)

		return action


	def reward(self, nodes=None, origin=None, action=None):
		""" Calculates the reward of an action in a state R(s,a)

		Parameters
		----------

		Returns
		-------
		"""
		# U(s,a) = r_u log10( 1 + wL + wO )
		
		# R(s,a) = U(s,a) - (D(s,a) + O(s,a))
		return (U - (D + O))

	def update(self):
		""" Updates a Q value, based on a state-action pair

		Parameters
		----------

		Returns
		-------
		"""
		pass