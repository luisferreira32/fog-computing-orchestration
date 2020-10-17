import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from algorithms.qlearning import Qlearning, statetuple
from algorithms.basic import RandomAlgorithm
from fog.node import Core
from fog.coms import Task
from fog import configs
from tools import utils

# -- aux functions

def assert_state(nodes, nL):
	Qs = []
	for n in nodes: Qs.append(n.qs())
	return [nL, len(nL.w), Qs]

def base_setup():
	nodes = [Core("n0",0), Core("n1",1), Core("n2",2), Core("n3",3), Core("n4",4)]
	# set up a offload state (nL, 1, [1,0,0])
	nodes[0].recieve(Task(0))
	nodes[0].queue(Task(0))

	return nodes

# -- test functions

def test_possible_actions():
	nodes = base_setup()
	ql = Qlearning(epsilon=1)
	state = assert_state(nodes, nodes[0])

	