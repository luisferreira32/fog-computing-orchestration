#!/usr/bin/env python

import numpy as np
from sim_env.configs import CPU_CLOCKS, RAM_SIZES
from sim_env.configs import BASE_SLICE_CHARS, TIME_STEP, PACKET_SIZE, RAM_UNIT, CPU_UNIT

def average_delays(case=BASE_SLICE_CHARS, time_step=TIME_STEP):
	# constants
	total_arrivals = round(sum(case["arrivals"]),2)
	constraints = [tp[0] for tp in case["task_type"]]
	cpu = [tp[1] for tp in case["task_type"]]
	ram = [tp[2] for tp in case["task_type"]]
	avg_constraint = round(np.mean(constraints), 2)
	avg_processing = round(1000*np.mean(cpu)*PACKET_SIZE/(CPU_UNIT*1e9),2)
	avg_mem = round(np.mean(ram)/RAM_UNIT,2)

	# prints
	print("On case with name", case["case"],":")
	print("Per timestep of size", 1000*time_step, "ms,", total_arrivals,"tasks arrive.")
	print("With an average of", avg_constraint,"ms time constraint,",avg_processing,"ms for processing a task and", avg_mem,"ram units required per task" )
	print("Resulting in an expected average delay of", avg_processing*total_arrivals/len(case["arrivals"]), "ms assuming each task can be processed whenever it arrives")
	print("A node has at least", CPU_CLOCKS[0]/CPU_UNIT,"cpu units and", RAM_SIZES[0]/RAM_UNIT,"ram units")
	print("---")
	print("In each buffer,")
	for tp,ar in zip(case["task_type"], case["arrivals"]):
		print("-")
		print(ar*10, "tasks arrive each", 1000*time_step*10, "ms")
		print(tp[0], "ms, is the time constraint")
		print(round(1000*tp[1]*PACKET_SIZE/(CPU_UNIT*1e9),2),"ms, to process a task")
		print(round(tp[2]/(RAM_UNIT),2),"ram units, to have a task in the buffer")

