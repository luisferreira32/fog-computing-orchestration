#!/usr/bin/env python

import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
    sys.path.append("/home/yourself/fog-computing-orchestration")

from sim_env.events import Event_queue, Task_arrival, Offload, Start_processing, Task_finished, Set_arrivals, Discard_task
from sim_env.core_classes import create_random_node, Task
from sim_env.configs import N_NODES, MAX_QUEUE

def test_set_arrivals():
	# couple nodes
	nodes = [create_random_node(i) for i in range(N_NODES)]
	# event queue
	evq = Event_queue()
	ev = Set_arrivals(0, 1, nodes)
	ev.execute(evq)
	# now there must be at least another set arrivals
	assert evq.queueSize() > 0
	# and at least one Task_arrival, since p=50%
	count = 0
	while evq.hasEvents():
		ev = evq.popEvent()
		if ev.classtype != "Task_arrival" and ev.classtype != "Set_arrivals":
			assert 1==0
		count += 1
	assert count > 0
	# and last event would be to set more tasks
	assert ev.classtype == "Set_arrivals"

def test_task_arrival():
	node = create_random_node(0)
	evq = Event_queue()
	# max queue is 10, so only 10 tasks should be accepted
	for i in range(MAX_QUEUE):
		assert Task_arrival(i, node, 0, Task(i)).execute(evq) == None
	t = Task(0)
	assert Task_arrival(0, node, 0, t).execute(evq) == t
	# but other MAX_QUEUE should be able to fill
	for i in range(MAX_QUEUE):
		assert Task_arrival(i, node, 1, Task(i)).execute(evq) == None
		assert Task_arrival(i, node, 2, Task(i)).execute(evq) == None
	assert Task_arrival(0, node, 1, t).execute(evq) == t
	assert Task_arrival(0, node, 2, t).execute(evq) == t

def test_task_finished():
	evq = Event_queue()
	node = create_random_node(0)
	tasks = [Task(i) for i in range(MAX_QUEUE+1)]
	for t in tasks:
		Task_arrival(t._timestamp, node, 0, t).execute(evq)
	buffer_size = len(node.buffers[0])
	# should return the tasks that are in the queue, with finished time, None for the last
	for t in tasks[:-1]:
		assert Task_finished(t._timestamp+1, node, 0, t).execute(evq) == t
		assert buffer_size-1==len(node.buffers[0])
		buffer_size -= 1
		assert t.task_delay() == 1
	assert Task_finished(tasks[-1]._timestamp+1, node, 0, tasks[-1]).execute(evq) == None

def test_start_processing():
	evq = Event_queue()
	node = create_random_node(0)
	t1 = Task(0,5,10, 400, 400) # 400 cycles/bit, 400 Mb
	t2 = Task(0,5,10, 500, 1200) # 500 cycles/bit, 1200 Mb
	node.add_task_on_slice(0, t1)
	node.add_task_on_slice(0, t2)
	# try to process 2 on slice 0
	Start_processing(0, node, 0, 2).execute(evq) 
	# at least one task started processing and it schedule a finishing event
	assert evq.queueSize() > 0
	assert t1.is_processing() == True
	assert t2.is_processing() == True
	evq.popEvent().execute(evq)
	evq.popEvent().execute(evq)
	assert t1.is_processing() == False
	assert t2.is_processing() == False

def test_offload():
	evq = Event_queue()
	n1 = create_random_node(0)
	n2 = create_random_node(1)
	n1.set_communication_rates([n1,n2])
	assert n1._communication_rates[n2.index] != 0
	t1 = Task(0,5,10, 400, 400) # 400 cycles/bit, 400 Mb
	n1.add_task_on_slice(0, t1)
	Offload(0, n1, 0, n2).execute(evq)
	# there is a task arrival event
	ev = evq.popEvent()
	assert ev.classtype == "Task_arrival"
	# had a comunication time, and arrives at an empty buffer
	assert ev.time > 0
	assert ev.execute(evq) == None

def test_discard_task():
	evq = Event_queue()
	node = create_random_node(0)
	# place three tasks in slice 0
	t1 = Task(0); t2 = Task(0); t3 = Task(0)
	Task_arrival(0, node, 0, t1).execute(evq)
	Task_arrival(0, node, 0, t2).execute(evq)
	Task_arrival(0, node, 0, t3).execute(evq)
	# start the processing of two
	assert node._avail_cpu_units == node.cpu_frequency
	Start_processing(0, node, 0, 2).execute(evq)
	assert node._avail_cpu_units == node.cpu_frequency-2
	# discard one
	assert Discard_task(0,node,0,t1).execute(evq) == t1
	assert t1.is_processing() == True
	assert t1.is_completed() == False
	assert Discard_task(0,node,0,t1).execute(evq)  == None # not on the queue anymore
	assert node._avail_cpu_units == node.cpu_frequency-1
	# if we discard one not processing
	assert Discard_task(0,node,0,t3).execute(evq)  == t3
	assert t3.is_processing() == False
	assert t3.is_completed() == False
	assert node._avail_cpu_units == node.cpu_frequency-1
	assert Discard_task(0,node,0,t3).execute(evq)  == None # not on the queue anymore