#!/usr/bin/env python

import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
    sys.path.append("/home/yourself/fog-computing-orchestration")

from sim_env.core_classes import Task
from sim_env.events import Task_arrival, Event_queue
from sim_env.configs import SIM_TIME_STEPS

def test_standard_sorting():
	eq = Event_queue()
	e1 = Task_arrival(0.1*SIM_TIME_STEPS, None, None, Task(0))
	e2 = Task_arrival(0.15*SIM_TIME_STEPS, None, None, Task(0))
	e3 = Task_arrival(0.16*SIM_TIME_STEPS, None, None, Task(0))

	# test standard sorting
	eq.addEvent(e1)
	eq.addEvent(e3)
	eq.addEvent(e2)
	assert eq.popEvent()==e1
	assert eq.popEvent()==e2
	assert eq.popEvent()==e3

def test_information_consistency():
	eq = Event_queue()
	# check information consistency
	e1 = Task_arrival(0.1*SIM_TIME_STEPS, None, None, Task(0))
	e4 = Task_arrival(0.2*SIM_TIME_STEPS, None, None, Task(0))
	eq.addEvent(e1)
	eq.addEvent(e4)
	assert eq.popEvent().time == 0.1*SIM_TIME_STEPS
	assert eq.popEvent().time == 0.2*SIM_TIME_STEPS

def test_inverted_sorting():
	eq = Event_queue()
	# check inverted sorting
	e1 = Task_arrival(0.1*SIM_TIME_STEPS, None, None, Task(0))
	e5 = Task_arrival(0.9*SIM_TIME_STEPS, None, None, Task(0))
	eq.addEvent(e5)
	eq.addEvent(e1)
	assert eq.popEvent() == e1
	assert eq.queueSize() == 1

def test_boundary_cases():
	eq = Event_queue()
	# check if boundaries are respected	
	e6 = Task_arrival(SIM_TIME_STEPS+1, None, None, Task(0))
	e7 = Task_arrival(-1, None, None, Task(-1))

	eq.addEvent(e6)
	eq.addEvent(e7)
	assert eq.queueSize() == 0

def test_double_check():
	eq = Event_queue()
	# test same values sorting
	e1 = Task_arrival(1, None, None, Task(0))
	e2 = Task_arrival(1, None, None, Task(0))
	e3 = Task_arrival(1, None, None, Task(0))
	e4 = Task_arrival(1, None, None, Task(0))
	e5 = Task_arrival(1, None, None, Task(0))
	eq.addEvent(e1)
	assert eq.popEvent() == e1
	eq.addEvent(e2)
	eq.addEvent(e3)
	eq.addEvent(e4)
	eq.addEvent(e5)
	assert eq.popEvent() == e2