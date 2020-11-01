#!/usr/bin/env python

import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
    sys.path.append("/home/yourself/fog-computing-orchestration")

import numpy as np

# imported test functions
from sim_env.core_classes import Task, task_processing_time, create_random_node, point_to_point_transmission_rate
from sim_env.configs import CPU_CLOCKS, RAM_SIZES

def test_task_processing_time():
	# t=0, size = 5 kb, delay = 10ms, cpu_demand = ? cycles/bit, memory = 400Mb
	t1 = Task(0,5000,10, 400, 400) # ? = 400
	t2 = Task(0,5000,10, 500, 400) # ? = 500
	t3 = Task(0,5000,10, 600, 400) # ? = 600
	# can't assert a processing time of a task that doesn't have cores working on it
	assert task_processing_time(t1) == "task not processing"
	t1.start_processing(1,1)
	t2.start_processing(1,1)
	t3.start_processing(2,1)
	assert task_processing_time(t1) == 0.002
	assert task_processing_time(t1) < task_processing_time(t2)
	assert task_processing_time(t1) > task_processing_time(t3)
	# and check out the None
	assert task_processing_time() == "no task given"

def test_create_random_node():
	nodes = [create_random_node(i) for i in range(5)]
	for n1 in nodes:
		for n2 in nodes:
			if n1 == n2: continue
			assert n1.index != n2.index
			# can't be in the same place by default
			assert np.any([n1.x != n2.x, n1.y != n2.y])
		assert n1.cpu_frequency >= min(CPU_CLOCKS)
		assert n1.cpu_frequency <= max(CPU_CLOCKS)
		assert n1.ram_size >= min(RAM_SIZES)
		assert n1.ram_size <= max(RAM_SIZES)

def test_point_to_point_transmission_rate():
	n1 = create_random_node()
	n2 = create_random_node(n1.index+1)
	assert point_to_point_transmission_rate(n1, n2) > 0
	assert point_to_point_transmission_rate(n1, n1) == 0

