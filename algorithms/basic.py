# imports from our utils and modules
from fog import node
from fog import configs
from tools import utils

def randomalgorithm(state, nodes):
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
		n0 = nodes[nO_index]
		#  unload a random within the possible
		w0 = round(utils.uniformRandom(w))
	else:
		w0 = 0
		n0 = None

	return [w0, n0]

def leastqueue(state, nodes):
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
	n0 = None
	if possible_nO: nO_index = possible_nO[0]	
	for nO in possible_nO:
		if Qsizes[nO] < Qsizes[nO_index]:
			nO_index=nO
			n0 = nodes[nO]
	
	w0 = 0
	if n0 is not None: w0 = round(utils.uniformRandom(w))

	return [w0, n0]


def nearestnode(state, nodes):
	"""Offloads tasks to the node with the minimum distance to this one, and space on queue
	"""
	# unpack it
	nL = state[0]
	w = state[1]
	Qsizes = state[2]

	# send to nearest with a lesser queue
	e0 = 9999999
	n0 = None
	for n,e in nL.comtime.items():
		if n.qs() >= nL.qs(): continue
		if e < e0:
			n0 = n
			e0 = e
	
	w0 = 0
	if n0 is not None: w0 = round(utils.uniformRandom(w))

	return [w0, n0]