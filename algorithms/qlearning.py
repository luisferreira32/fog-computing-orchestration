# external imports
import math
import random
import copy

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

	def changeiter(self, epsilon=None, ar=None,sr=None):
		if epsilon is not None: self.epsilon = epsilon
		if sr is not None: self.sr=sr
		if ar is not None: self.ar=ar

	def addstate(self, state=None, nodes=None):

		if nodes is None or state is None:
			return -1

		# if the new key is not in the qtable
		if state not in self.qtable:
			# create the dict of q zeroed actions
			actions = {}
			for w0 in range(0,configs.MAX_W+1):
				for n in nodes:
					if n.name != state[0]:
						actionkey = actiontuple(nO=n,w0=w0)
						actions[actionkey] = 0
			self.qtable[state] = actions
		else:
			return 1

		return 0

	def execute(self, nL=None, nodes=None):

		action = [0, None]
		if nodes is None or nL is None:
			return action

		state = statetuple(nodes, nL)
		self.addstate(state, nodes)

		# can only offload for a lesser queue
		possibleactions = []
		possibleactionsq = []
		for act, qvalue in self.qtable[state].items():
			(w0, nO) = act
			if w0 == 0: # if there is no offload, nodes are eligble
				possibleactions.append(act)
				possibleactionsq.append(qvalue)
				continue
			if w0 > len(nL.w): continue
			queues = dict(state[2])
			if queues[nO] > nL.qs(): continue
			possibleactions.append(act)
			possibleactionsq.append(qvalue)

		if not possibleactions:
			return action

		x = random.random()
		# explore
		if x < self.epsilon:
			(w0, nO) = random.choice(possibleactions)
		# exploit
		else:
			i = possibleactionsq.index(max(possibleactionsq))
			(w0, nO) = possibleactions[i]

		# find the node based on the name 
		for n in nodes:
			if n.name == nO:
				destnode = n

		action = [w0, destnode]
		return action


	def qreward(self, nL, action=None):
		""" Calculates an instant reward based on an action taken
		"""

		if action is None:
			print("[Q DEBUG] Failed to calculate reward, returning zero.")
			return 0

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

	def update(self, state=None, action=None, nextstate=None, reward=0, nodes=None):
		""" Updates a q-table entry based on a state transition and instant reward
		"""

		if state is None or action is None or nextstate is None or nodes is None:
			print("[Q DEBUG] Invalid parameters to update q table.")
			return None
		# add new states that weren't yet visited
		self.addstate(nextstate, nodes)
		# action key doesn't include nL, since that's in the state
		actionkey = actiontuple(action)

		newQ = (1-self.alpha)*self.qtable[state][actionkey] + self.alpha*(reward 
			+ self.discount_factor*max(self.qtable[nextstate].values()))
		self.qtable[state][actionkey] = newQ
		return newQ


	# ----------------------------------- DISPLAY FUNCTIONS ----------------------------------------------
	def printQtable(self):
		for state, actions in self.qtable.items():
			print("State",state)
			for action, qvalue in actions.items():
				print("Action", action, "qvalue",qvalue)

	def printQtableNZ(self):
		for state, actions in self.qtable.items():
			print("State",state)
			for action, qvalue in actions.items():
				if qvalue != 0:	print("Action", action, "qvalue",qvalue)


# ----------------------------------- AUXILIARY FUNCTIONS ----------------------------------------------

def statetuple(nodes=None, nL=None):
	""" Creates a state tuple given nodes
	"""
	auxq = {}
	for n in nodes:
		auxq[n.name] = n.qs()
	state = tuple([nL.name, len(nL.w), frozenset(auxq.items())])
	return copy.deepcopy(state)

def actiontuple(action=None, nO=None, w0=None):
	""" Creates an action tuple (key) given an action list
	"""
	if action:
		(w0, nO) = action
		actionkey = tuple([w0, nO.name])
		return copy.deepcopy(actionkey)
	elif nO is not None and w0 is not None:
		actionkey = tuple([w0, nO.name])
		return copy.deepcopy(actionkey)
	return None