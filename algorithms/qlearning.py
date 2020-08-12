# external imports
import math

# our module imports
from fog import configs

class Qlearning(object):
	"""
	The object of Qlearning, containing all methods and attributes necessary to train, and run a q learning algorithm
	
	...

	Attributes
	----------
	qtable : dict[][]
		a dictionary between state and action with Q-values ready to be trained


	Methods
	-------
	execute()
		chooses a set of actions for this timestep
	update()
		updates the q values from the table
	"""

	def __init__(self, nodes=None, max_q=configs.MAX_QUEUE, max_influx=configs.MAX_INFLUX):
		for 