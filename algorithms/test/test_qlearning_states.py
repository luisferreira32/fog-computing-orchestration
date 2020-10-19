import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from algorithms.qlearning import Qlearning, statetuple
from fog.node import Core
from fog.coms import Task
from fog import configs


def assert_state(nodes, nL):
	Qs = []
	for n in nodes: Qs.append(n.qs())
	return [nL, len(nL.w), Qs]

def test_state_creation():
	nodes = [Core("n0",0), Core("n1",1), Core("n2",2)]
	ql = Qlearning()
	ql.setnodes(nodes)

	# add minumum state (nL, 1, [0,0])
	nodes[0].recieve(Task(0))
	state = assert_state(nodes, nodes[0])

	# no states yet then addeed a new one if we visit it
	assert len(ql.qtable) == 0
	ql.addstate(state)
	assert len(ql.qtable) == 1
	statekey = statetuple(state)
	assert len(ql.qtable[statekey]) == (configs.MAX_W)*2
