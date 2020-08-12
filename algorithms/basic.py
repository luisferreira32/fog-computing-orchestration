# external imports
import random
random.seed(1)

# imports from our utils and modules
from fog import node
from fog import configs

def randomalgorithm(origin, nodes, recieving):
	"""Gives a random action based on the state
	
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
	e = 0
	if origin.excessinflux(recieved=recieving) > 0:
		# if there is excess, offload the excess and not a random thing
		e = origin.excessinflux(recieved=recieving)

	actions = []
	# randomly offload decisions
	while e >= 1:
		# choose a different node
		randomoff = random.choice(nodes)
		r = int(random.random()*e)+1
		if randomoff == origin:
			continue 
		# and a random quantity to offload to that node
		e -= r
		actions.append([origin, randomoff, r])
	return actions

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
	actions = []
	queues = []
	moves = {}
	for n in nodes:
		queues.append(n.qs())
		moves[n] = 0

	e = 0
	if origin.excessinflux(recieved=recieving) > 0:
		e = origin.excessinflux(recieved=recieving)

	while e >= 1:
		i = queues.index(min(queues))
		if nodes[i] != origin:
			moves[nodes[i]] +=1
		queues[i] +=1
		e -= 1
	for n in nodes:
		if n != origin and moves[n] != 0:
			actions.append([origin, n, moves[n]])
	
	return actions

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
	actions = []
	queues = {}
	moves = {}
	distances = {}
	for n in nodes:
		if n == origin: continue
		queues[n] = n.qs()
		moves[n] = 0
		distances[n] = node.distance(origin,n)

	e = 0
	if origin.excessinflux(recieved=recieving) > 0:
		e = origin.excessinflux(recieved=recieving)

	while e >= 1 and distances:
		n = min(distances, key=distances.get)
		if queues[n] >= configs.MAX_QUEUE:
			distances.pop(n)
			continue
		moves[n] += 1
		queues[n] += 1
		e -= 1
	for n in nodes:
		if n != origin and moves[n] != 0:
			actions.append([origin, n, moves[n]])
	
	return actions