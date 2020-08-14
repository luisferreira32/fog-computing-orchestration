# external imports
import math
import random

# our module imports
from fog import configs
from fog import coms
from fog import node

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
	qreward()
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

		return action


	def qreward(self, nodes=None, action=None, r12=None, sr=configs.SERVICE_RATE):
		""" Calculates the reward of an action in a state R(s,a), and some pre-calculated constants

		Parameters
		----------

		Returns
		-------
		"""
		if nodes is None or action is None or r12 is None:
			print("[Q DEBUG] Failed to calculate reward, returning zero.")
			return 0

		(origin, dest, w0) = action

		# U(s,a) = r_u log( 1 + wL + w0 )  --- log2 ? they don't specify
		Usa = self.r_utility * math.log2( 1 + origin.wL + w0)

		# t_w = QL/uL if(wL!=0) + (QL/uL + Q0/u0) if(w0!=0)
		t_w = node.wtime(origin, dest, w0)
		# t_c = 2*T*wO / r_LO
		t_c = coms.comtime(w0, r12)
		# t_e = I*CPI*wL/f_L + I*CPI*w0/f_0
		t_e = node.extime(origin, dest, w0)

		# D(s,a) = x_d * (t_w + t_c + t_e)/(wL + w0)
		if origin.wL+w0 != 0:
			Dsa = self.x_delay * (t_w+t_c+t_e)/(origin.wL + w0)
		else:
			Dsa = 0

		# P_overload,i = max(0, y_i - (Q_i,max - Q'_i))/ y_i
		# Q'_i = min(max(0, Q_i - sr_i) + w_i, Q_i,max)
		Q_nL = min(max(0,origin.qs()-sr)+origin.influx, configs.MAX_QUEUE)
		if origin.influx != 0:
			P_oL = max(0, origin.influx - (configs.MAX_QUEUE - Q_nL)) / origin.influx
		else:
			P_oL = 0

		Q_n0 = min(max(0,dest.qs()-sr)+dest.influx, configs.MAX_QUEUE)
		if dest.influx != 0:
			P_o0 = max(0, dest.influx - (configs.MAX_QUEUE - Q_n0)) / dest.influx
		else:
			P_o0 = 0

		# O(s,a) = x_o * (wL * P_overload,L + w0 * P_overload,0)/(wL + w0)
		if origin.wL+w0 != 0:
			Osa = self.x_overload * (origin.wL*P_oL + w0*P_o0)/(origin.wL + w0)
		else:
			Osa = 0
		
		# R(s,a) = U(s,a) - (D(s,a) + O(s,a))
		return (Usa - (Dsa + Osa))

	def update(self):
		""" Updates a Q value, based on a state-action pair

		Parameters
		----------

		Returns
		-------
		"""
		pass