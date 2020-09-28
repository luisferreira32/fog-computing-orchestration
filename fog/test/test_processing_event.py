import sys
sys.path.append("/home/yourself/fog-computing-orchestration")

from fog import events
from fog import node
from fog import configs
from fog import coms


def test_empty_cpu_queue():
	# it should not generate more events and return nothing
	eq = events.EventQueue()
	n1 = node.Core()
	e1 = events.Processing(0,n1)

	assert e1.classtype == "Processing"
	assert eq.queueSize() == 0
	assert e1.execute(eq) == None
	assert eq.queueSize() == 0


def test_one_task_cpu_queue():
	eq = events.EventQueue()
	n1 = node.Core()
	t1 = coms.Task(0)
	assert n1.queue(t1) == None
	e1 = events.Processing(0,n1)

	# returns the first task with it's set processing time and generates 1 event
	assert e1.execute(eq) == t1
	assert t1.delay == t1.il*n1.cpi/n1.cps
	assert eq.queueSize() == 1
	# the node n1 is processing!
	assert n1.processing
	# the empty processing that was generated
	e2 = eq.popEvent()
	assert e2.execute(eq) == None
	assert not n1.processing
	assert eq.queueSize() == 0


def test_multi_task_cpu_queue():
	eq = events.EventQueue()
	n1 = node.Core()
	t1 = coms.Task(0)

	# get the queue full
	for i in range(10):
		assert n1.queue(t1) == None
	assert n1.queue(t1) == t1

	eq.addEvent(events.Processing(0,n1))
	ptime = 0
	while eq.hasEvents():
		e1 = eq.popEvent()
		t = e1.execute(eq)
		ptime += t1.il*n1.cpi/n1.cps
		if t is None:
			assert not n1.processing
		else:
			assert t == t1
			assert t.delay == ptime


def test_multi_cpu_one_tast():
	# should queue one event to stop it from processing after
	eq = events.EventQueue()
	n1 = node.Core()
	n2 = node.Core()

	t1 = coms.Task(0)
	# to finish first
	t2 = coms.Task(0, instruction_lines=configs.DEFAULT_IL - 10)

	n1.queue(t1)
	eq.addEvent(events.Processing(0,n1))
	n2.queue(t2)
	eq.addEvent(events.Processing(0,n2))

	assert n1.processing == False
	ev = eq.popEvent()
	assert t1 == ev.execute(eq)
	assert n1.processing == True
	ev = eq.popEvent()
	assert t2 == ev.execute(eq)
	ev = eq.popEvent()
	assert ev.pn == n2
	ev.execute(eq)
	ev = eq.popEvent()
	assert ev.pn == n1
	assert eq.hasEvents() == False
	assert ev.execute(eq) == None
	assert n1.processing == False