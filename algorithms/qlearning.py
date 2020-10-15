# external imports
import math
import copy

# our module imports
from fog import configs
from fog import coms
from fog import node
from tools import utils

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

	def __init__(self, a=0.5, df=0.5, sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE):
		# init with an empty dict table
		self.qtable = {}
		self.alpha = a
		self.discount_factor = df
		self.r_utility = 10
		self.x_delay = 1
		self.x_overload = 150
		self.epsilon = 0.9
		self.sr=sr
		self.ar=ar

		# for the sim specific variables
		self.nodes = None
		self.updatable = True

	def __str__(self):
		return "QL [r_u: "+str(self.r_utility)+" x_d: "+str(self.x_delay)+" x_o: "+str(self.x_overload)+"]"

	# -- for start up ---

	def changeiter(self, epsilon=None, ar=None,sr=None):
		if epsilon is not None: self.epsilon = epsilon
		if sr is not None: self.sr=sr
		if ar is not None: self.ar=ar

	def setnodes(self,nodes):
		self.nodes = nodes

	# --- q-table related ---

	def addstate(self, state=None):
		state = statetuple(state)

		# if the new key is not in the qtable
		if state not in self.qtable:
			# create the dict of q zeroed actions
			actions = {}
			for w0 in range(1,configs.MAX_W+1):
				for n in self.nodes:
					if n != state[0]:
						actionkey = actiontuple(w0=w0, nO=n)
						actions[actionkey] = 0
			self.qtable[state] = actions
		else:
			return 1

		return 0

	def execute(self, state = None):

		action = [0, None]
		if state is None:
			return action

		(nL, w, Qs) = state
		Qs = list(Qs)
		self.addstate(state)
		state = statetuple(state)

		# can only offload for a lesser queue
		possibleactions = []
		possibleactionsq = []
		for act, qvalue in self.qtable[state].items():
			(w0, nO) = act
			if w0 > len(nL.w): continue
			if Qs[nO.index] > nL.qs(): continue
			possibleactions.append(act)
			possibleactionsq.append(qvalue)

		if not possibleactions:
			return action

		x = utils.uniformRandom()
		# explore
		if x < self.epsilon:
			(w0, nO) = utils.randomChoice(possibleactions)
		# exploit
		else:
			i = possibleactionsq.index(max(possibleactionsq))
			(w0, nO) = possibleactions[i]

		return [w0, nO]

	def update(self, state=None, action=None, nextstate=None, reward=0, nodes=None):
		""" Updates a q-table entry based on a state transition and instant reward
		"""

		if state is None or action is None or nextstate is None or nodes is None:
			print("[Q DEBUG] Invalid parameters to update q table.")
			return None
		# add new states that weren't yet visited
		self.addstate(nextstate)
		# and tuple them to do keys
		statekey = statetuple(state)
		actionkey = actiontuple(action)
		nextstatekey = statetuple(nextstate)

		newQ = (1-self.alpha)*self.qtable[statekey][actionkey] + self.alpha*(reward 
			+ self.discount_factor*max(self.qtable[nextstatekey].values()))
		self.qtable[statekey][actionkey] = newQ
		return newQ


	def reward(self, state=None, action=None):
		""" Calculates an instant reward based on an action taken
		"""

		if action is None or state is None:
			print("[Q DEBUG] Failed to calculate reward, returning zero.")
			return 0

		nL= state[0]
		(w0, nO) = action

		# U(s,a) = r_u log( 1 + wL + w0 )  --- log2 ? they don't specify
		wL = min(configs.MAX_QUEUE-nL.qs(), len(nL.w) - w0)
		Usa = self.r_utility * math.log2( 1 + wL + w0)

		# t_w = QL/uL if(wL!=0) + (QL/uL + Q0/u0) if(w0!=0)
		t_w = node.wtime(nL, nO, wL, w0)
		# t_c = 2*T*wO / r_LO
		t_c = w0*nL.comtime[nO]
		# t_e = I*CPI*wL/f_L + I*CPI*w0/f_0
		t_e = node.extime(nL, nO, wL, w0)

		# D(s,a) = x_d * (t_w + t_c + t_e)/(wL + w0)
		if wL+w0 != 0:
			Dsa = self.x_delay * (t_w+t_c+t_e)/(wL + w0)
		else:
			Dsa = 0

		# P_overload,i = max(0, y_i - (Q_i,max - Q'_i))/ y_i
		# Q'_i = min(max(0, Q_i - sr_i) + w_i, Q_i,max)
		Q_nL = min(max(0,nL.qs()-self.sr)+len(nL.w), configs.MAX_QUEUE)
		if self.ar/configs.N_NODES != 0:
			P_oL = max(0, (self.ar/configs.N_NODES) - (configs.MAX_QUEUE - Q_nL)) / (self.ar/configs.N_NODES)
		else:
			P_oL = 0

		Q_n0 = min(max(0,nO.qs()-self.sr)+len(nO.w), configs.MAX_QUEUE)
		if (self.ar/configs.N_NODES) != 0:
			P_o0 = max(0, (self.ar/configs.N_NODES) - (configs.MAX_QUEUE - Q_n0)) / (self.ar/configs.N_NODES)
		else:
			P_o0 = 0

		# O(s,a) = x_o * (wL * P_overload,L + w0 * P_overload,0)/(wL + w0)
		if wL+w0 != 0:
			Osa = self.x_overload * (wL*P_oL + w0*P_o0)/(wL + w0)
		else:
			Osa = 0
		
		# R(s,a) = U(s,a) - (D(s,a) + O(s,a))
		return (Usa - (Dsa + Osa))


	# ----------------------------------- DISPLAY FUNCTIONS ----------------------------------------------
	def printQtable(self):
		for state, actions in self.qtable.items():
			print("State",state[0].name, state[1:])
			for action, qvalue in actions.items():
				print("Action", action[0],action[1].name, "qvalue",qvalue)

	def printQtableNZ(self):
		for state, actions in self.qtable.items():
			for action, qvalue in actions.items():
				if qvalue != 0:
					print("State",state[0].name, state[1:])
					print("Action",  action[0],action[1].name, "qvalue",qvalue)

	def printQtablePV(self):
		for state, actions in self.qtable.items():
			for action, qvalue in actions.items():
				if qvalue > 0:
					print("State",state[0].name, state[1:])
					print("Action",  action[0],action[1].name, "qvalue",qvalue)


# ----------------------------------- AUXILIARY FUNCTIONS ----------------------------------------------

def statetuple(state=None,nodes=None, nL=None):
	""" Creates a state tuple given nodes
	"""
	if state is not None:
		if isinstance(state[2], list):
			state[2] = tuple(state[2])
		return tuple(state)
	elif nodes is not None and nL is not None:
		Qs = []
		for n in nodes:	Qs.append(n.qs())
		return tuple([nL.name, len(nL.w), tuple(Qs)])
	return None

def actiontuple(action=None, nO=None, w0=None):
	""" Creates an action tuple (key) given an action list
	"""
	if action:
		return tuple(action)
	elif nO is not None and w0 is not None:
		return tuple([w0, nO])
	return None