import sys
sys.path.append("/home/yourself/fog-computing-orchestration")

from fog import events
from fog import node
from fog import configs
from fog import coms

def test_one_recieve():
	evq = events.EventQueue()
	n1 = node.Core()
	t1 = coms.Task(0)
	e1 = events.Recieving(0,n1,t1, ar=5, nodes=[n1])
	e1.execute(evq)
	assert evq.queueSize() == 1
	assert n1.decide() == t1