import sys
sys.path.append("/home/yourself/fog-computing-orchestration")

from fog import events

def test_standard_sorting():
	eq = events.EventQueue()
	e1 = events.Event(1)
	e2 = events.Event(1.5)
	e3 = events.Event(1.6)

	# test standard sorting
	eq.addEvent(e1)
	eq.addEvent(e3)
	eq.addEvent(e2)
	assert eq.popEvent()==e1
	assert eq.popEvent()==e2
	assert eq.popEvent()==e3

def test_information_consistency():
	eq = events.EventQueue()
	# check information consistency
	e1 = events.Event(1)
	e4 = events.Event(2)
	eq.addEvent(e1)
	eq.addEvent(e4)
	assert eq.popEvent().time == 1
	assert eq.popEvent().time == 2

def test_inverted_sorting():
	eq = events.EventQueue()
	# check inverted sorting
	e1 = events.Event(1)
	e5 = events.Event(10)

	eq.addEvent(e5)
	eq.addEvent(e1)
	assert eq.popEvent() == e1
	assert eq.queueSize() == 1

def test_boundary_cases():
	eq = events.EventQueue()
	# check if boundaries are respected	
	e6 = events.Event(10000)
	e7 = events.Event(-1)

	eq.addEvent(e6)
	eq.addEvent(e7)
	assert eq.queueSize() == 0

def test_double_check():
	eq = events.EventQueue()
	# test same values sorting
	e1 = events.Event(1,"one")
	e2 = events.Event(1,"two")
	e3 = events.Event(1,"three")
	e4 = events.Event(1,"four")
	e5 = events.Event(1,"five")
	eq.addEvent(e1)
	assert eq.popEvent() == e1
	eq.addEvent(e2)
	eq.addEvent(e3)
	eq.addEvent(e4)
	eq.addEvent(e5)
	assert eq.popEvent() == e2
