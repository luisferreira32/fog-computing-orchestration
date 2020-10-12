import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from fog import events
from fog import node
from fog import configs
from fog import coms

def test_send_recieve_one():
	evq = events.EventQueue()
	n1 = node.Core()
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
	assert e.it == t1
	e.execute(evq)
	assert n2.emptyqueue() == False
	assert evq.popEvent().execute(evq) == t1
	assert t1.timestamp == 0