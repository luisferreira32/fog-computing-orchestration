# external imports
import random
random.seed(17)

# imports from our utils and modules
from fog import node
from fog import configs

def randomalgorithm(origin, nodes, recieving):
	"""Gives a random action based on the state and possible actions
	
	Parameters
	----------
	origin
		is the node state being processed, i.e. the one recieving allocations
	nodes
		the state of the fog, with queues and everything
	recieving
		state of coms (counts as an extra to influx)

	Return
	------
	a set of actions to take by this node in this state
	"""	
	e = int(random.random()*origin.influx)
	action = []
	if origin.excessinflux(recieved=recieving) > e:
		# if there is excess, offload the excess and not a random thing
		e = origin.excessinflux(recieved=recieving)
	# choose a different node
	randomoff = random.choice(nodes)
	if randomoff != origin:
		action = [origin, randomoff, e]		

	return action

def leastqueue(origin, nodes, recieving):
	"""Offloads tasks to the node with the minimum queue status
	
	Parameters
	----------
	origin
		is the node state being processed, i.e. the one recieving allocations
	nodes
		the state of the fog, with queues and everything
	recieving
		state of coms (counts as an extra to influx)

	Return
	------
	a set of actions to take by this node in this state
	"""
	queues = {}
	for n in nodes:
		if n == origin: continue
		queues[n] = n.qs()

	action = []
	e = 0
	if origin.excessinflux(recieved=recieving) > e:
		e = origin.excessinflux(recieved=recieving)
	n = min(queues, key=queues.get)
	action = [origin, n, min(e,configs.MAX_QUEUE-queues[n])]
	
	return action

def nearestnode(origin, nodes, recieving):
	"""Offloads tasks to the node with the minimum distance to this one, and space on queue
	
	Parameters
	----------
	origin
		is the node state being processed, i.e. the one recieving allocations
	nodes
		the state of the fog, with queues and everything
	recieving
		state of coms (counts as an extra to influx)

	Return
	------
	a set of actions to take by this node in this state
	"""
	queues = {}
	distances = {}
	for n in nodes:
		if n == origin: continue
		queues[n] = n.qs()
		distances[n] = node.distance(origin,n)

	e = 0
	action = []
	if origin.excessinflux(recieved=recieving) > e:
		e = origin.excessinflux(recieved=recieving)

	while distances:
		n = min(distances, key=distances.get)
		if queues[n] >= configs.MAX_QUEUE:
			distances.pop(n)
			continue
		break
	if distances:
		action = [origin, n, min(e,configs.MAX_QUEUE-queues[n])]
	
	return action