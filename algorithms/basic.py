# external imports
import random
random.seed(1)

def randomalgorithm(node, nodes, recieving):
	"""Gives a random action based on the state
	
	Parameters
	----------
	node
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
	if node.excessinflux(recieved=recieving) > 0:
		# if there is excess, offload the excess and not a random thing
		e = node.excessinflux(recieved=recieving)

	actions = []
	# randomly offload decisions
	while e > 0:
		# choose a different node
		randomoff = node
		while randomoff == node:
			randomoff = random.choice(nodes)
		# and a random quantity to offload to that node
		er = int(random.random()*e)+1
		e -= er
		actions.append([node, randomoff, er])
	return actions

def leastqueue(node, nodes, recieving):
	"""Offloads tasks to the node with the minimum queue status
	
	Parameters
	----------
	node
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
	for n in nodes:
		queues.append(n.qs())

	e = 0
	if node.excessinflux(recieved=recieving) > 0:
		e = node.excessinflux(recieved=recieving)

	while e >= 1:
		i = queues.index(min(queues))
		if nodes[i] != node:
			actions.append([node, nodes[i], 1])
		queues[i] +=1
		e -= 1
	
	return actions