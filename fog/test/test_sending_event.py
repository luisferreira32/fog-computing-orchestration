import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from fog import events
from fog import node
from fog import configs
from fog import coms

def test_one_send():
	evq = events.EventQueue()
	n1 = node.Core(placement=(0,0))
	n2 = node.Core(placement=(0,1))
	n1.setcomtime([n1, n2])
	n2.setcomtime([n1, n2])
	t1 = coms.Task(0)

	n1.send(t1, n2)
	e1 = events.Sending(0, n1)
	assert evq.hasEvents() == False
	evq.addEvent(e1)
	e = evq.popEvent()
	assert e == e1
	e.execute(evq)
	assert evq.hasEvents() == True
	e = evq.popEvent()
	assert e.classtype == "Recieving"
	assert e.time == n1.comtime[n2]

def test_impossible_send():
	evq = events.EventQueue()
	n1 = node.Core(placement=(0,0))
	n2 = node.Core(placement=(0,0))
	n1.setcomtime([n1, n2])
	n2.setcomtime([n1, n2])
	t1 = coms.Task(0)
	n1.send(t1,n2)
	e1 = events.Sending(0,n1)
	assert e1.execute(evq) == 1 # discarded 1
	assert evq.hasEvents() == False

def test_sending_arrival():
	evq = events.EventQueue()
	n1 = node.Core(placement=(0,0))
	n2 = node.Core(placement=(0,1))
	n3 = node.Core(placement=(0,3))
	n1.setcomtime([n1, n2, n3])
	n2.setcomtime([n1, n2, n3])
	n3.setcomtime([n1, n2, n3])
	t1 = coms.Task(0)
	t2 = coms.Task(0)

	# send two tasks at 0, to two different nodes
	n1.send(t1, n3)
	e1 = events.Sending(0, n1)
	e1.execute(evq)
	# force transmitting, since antena should be busy
	n1.transmitting = False
	n1.send(t2, n2)
	e2 = events.Sending(0, n1)
	e2.execute(evq)
	assert evq.hasEvents() == True
	e3 = evq.popEvent()
	e4 = evq.popEvent()
	# from n1 to n2 is faster than from n1 to n3
	assert e3.it == t2
	assert e4.it == t1
	assert e3.time < e4.time