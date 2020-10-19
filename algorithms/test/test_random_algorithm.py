import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from algorithms.basic import RandomAlgorithm
from fog.node import Core
from fog.coms import Task

# quick test on set nodes
def test_rd_setnodes():
	# the first objects
	nodes = return_nodes()
	r = RandomAlgorithm()
	r.setnodes(nodes)
	assert r.nodes == nodes
	# another copy, BUT not the ones from the sim
	r.setnodes(return_nodes())
	assert r.nodes != nodes

def return_nodes():
	return [Core("n0"), Core("n1")]

# with one task Queue and three empty nodes, see unform choosing of distribution for 1 incoming task
def test_rd_simple():
	n1 = Core(index=0)
	n2 = Core(index=1)
	n3 = Core(index=2)
	n4 = Core(placement=[10, 10], index=3)
	nodes = [n1, n2, n3, n4]
	r = RandomAlgorithm(nodes)

	# counters
	n2c = 0
	n3c = 0
	n4c = 0

	n1.queue(Task(0))
	n1.recieve(Task(1))
	assert len(n1.w) == 1
	state = (n1, len(n1.w), [n1.qs(), n2.qs(), n3.qs(), n4.qs()])

	for x in range(10000):
		(w0, n0) = r.execute(state)
		if n0 == n2:
			n2c +=1
		elif n0 == n3:
			n3c +=1
		elif n0 == n4:
			n4c +=1
		else:
			# make sure it only offloads to possible destinations, never to himself
			assert 0==1

	totalnc = n2c + n3c + n4c
	assert round(n2c/totalnc,1) == 0.3
	assert round(n3c/totalnc,1) == 0.3
	assert round(n4c/totalnc,1) == 0.3

# if it doesn't offload to really busy nodes
def test_rd_busy():
	n1 = Core(index=0)
	n2 = Core(index=1)
	n3 = Core(index=2)
	n4 = Core(placement=[10, 10], index=3)
	nodes = [n1, n2, n3, n4]
	r = RandomAlgorithm(nodes)

	# counters
	n2c = 0
	n3c = 0
	n4c = 0

	n1.queue(Task(0))
	n1.recieve(Task(1))
	n2.queue(Task(0))
	n2.queue(Task(0))
	assert len(n1.w) == 1
	state = (n1, len(n1.w), [n1.qs(), n2.qs(), n3.qs(), n4.qs()])

	for x in range(10000):
		(w0, n0) = r.execute(state)
		if n0 == n2:
			n2c +=1
		elif n0 == n3:
			n3c +=1
		elif n0 == n4:
			n4c +=1
		else:
			assert 0==1

	totalnc = n2c + n3c + n4c
	assert round(n2c/totalnc,1) == 0.0
	assert round(n3c/totalnc,1) == 0.5
	assert round(n4c/totalnc,1) == 0.5

# and if it can refuse offloadings
def test_rd_no_offload():
	n1 = Core(index=0)
	n2 = Core(index=1)
	r = RandomAlgorithm([n1,n2])
	n1.recieve(Task(0))
	n2.queue(Task(0))
	state = (n1, len(n1.w), [0, 1])
	(w0, n0) = r.execute(state)
	assert n0 == None
	assert w0 == 0
