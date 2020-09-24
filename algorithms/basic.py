# imports from our utils and modules
from fog import node
from fog import configs
from tools import utils

def randomalgorithm(state):
	"""Gives a random action based on the state and limited possible actions
	"""	
	# unpack it
	nL = state[0]
	w = state[1]
	Qsizes = state[2]

	# check possible acitons
	possible_nO = []
	for i in range(0, len(Qsizes)):
		if Qsizes[i] < nL.qs(): possible_nO.append(i)
	if possible_nO:
		nO_index = utils.randomChoice(possible_nO)
		#  unload as much as you can
		w0 = configs.MAX_QUEUE-Qsizes[nO_index]
	else:
		w0 = 0
		nO_index = 0

	return [w0, nO_index]

def leastqueue(state):
	"""Offloads tasks to the node with the minimum queue status
	"""
	# unpack it
	nL = state[0]
	w = state[1]
	Qsizes = state[2]

	# check possible acitons
	possible_nO = []
	for i in range(0, len(Qsizes)):
		if Qsizes[i] < nL.qs(): possible_nO.append(i)
	nO_index = 0
	if possible_nO: nO_index = possible_nO[0]	
	for nO in possible_nO:
		if Qsizes[nO] < Qsizes[nO_index]: nO_index=nO
	# unload as much as you can
	w0 = 0
	if possible_nO: w0 = configs.MAX_QUEUE-Qsizes[nO_index]

	return [w0, nO_index]


def nearestnode(state):
	"""Offloads tasks to the node with the minimum distance to this one, and space on queue
	"""
	# unpack it
	nL = state[0]
	w = state[1]
	Qsizes = state[2]

	# send to nearest with a lesser queue
	e0 = None
	for n,e in nL.edges.items():
		if e.neigh.qs() >= nL.qs(): continue
		if e0 == None: e0 = e
		if e.comtime < e0.comtime: e0 = e
	# cheat to get to the index of the minimum
	nO_index = 0
	if e0 is not None: nO_index= int(e0.neigh.name[1])
	# unload as much as you can
	w0 = 0
	if e0 is not None: w0 = configs.MAX_QUEUE-Qsizes[nO_index]

	return [w0, nO_index]