import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from algorithms.qlearning import Qlearning
from fog.node import Core
from fog.coms import Task

# quick test on set nodes
def test_q_setnodes():
	# the first objects
	nodes = return_nodes()
	ql = Qlearning()
	ql.setnodes(nodes)
	assert ql.nodes == nodes
	# another copy, BUT not the ones from the sim
	ql.setnodes(return_nodes())
	assert ql.nodes != nodes

def return_nodes():
	return [Core("n0"), Core("n1")]

def test_q_iter():
	ql = Qlearning()
	assert ql.epsilon == 0.9
	ql.changeiter(epsilon =1)
	assert ql.epsilon == 1