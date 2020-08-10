# external imports
import random
random.seed(1)

def randomalgorithm(node, nodes, recieving):
	"""Gives a random action based on the state
	
	Parameters
	----------

	Return
	------
	"""
	e = int(random.random()*node.influx)
	if node.excessinflux(recieved=recieving) > e:
		# when i offload it's on this world clock time
		e += node.excessinflux(recieved=recieving)

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