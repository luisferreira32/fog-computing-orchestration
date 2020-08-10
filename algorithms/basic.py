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
	e = int(random.random()*node.influx)
	if node.excessinflux(recieved=recieving) > e:
		# when i offload it's on this world clock time
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
	minqnode = node
	e = node.influx
	while e >= 1:
		for n in nodes:
			if minqnode.qs() > n.qs():
				minqnode = n
		if minqnode != node:
			actions.append([node, minqnode, 1])
		e -= 1
	
	return actions