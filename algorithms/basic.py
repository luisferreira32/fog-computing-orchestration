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
	nO_index = utils.randomChoice(possible_nO)
	# w0 only has to be lower than the recieved w and the queue space
	w0 = utils.uniformRandom(min(w, configs.MAX_QUEUE-Qsizes[nO_index]))
	if w0 > 0: w0 = int(w0)

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
	nO_index = possible_nO[0]	
	for nO in possible_nO:
		if Qsizes[nO] < Qsizes[nO_index]: nO_index=nO
	# unload as much as you can
	w0 = min(w, configs.MAX_QUEUE-Qsizes[nO_index])

	return [w0, nO_index]

# TODO@ THIS ONE -- DEPEND ON HOW I'M GONNA FORMAT COMS... TEMPTED TO DO WIFI BROADCAST
def nearestnode(state):
	"""Offloads tasks to the node with the minimum distance to this one, and space on queue
	"""
	# unpack it
	nL = state[0]
	w = state[1]
	Qsizes = state[2]

	# check possible acitons
	possible_nO = []
	for i in range(0, len(Qsizes)):
		if Qsizes[i] < nL.qs(): possible_nO.append(i)
	nO_index = utils.randomChoice(possible_nO)
	# w0 only has to be lower than the recieved w and the queue space
	w0 = utils.uniformRandom(min(w, configs.MAX_QUEUE-Qsizes[nO_index]))

	return [w0, nO_index]