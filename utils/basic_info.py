#!/usr/bin/env python

import numpy as np
from sim_env.configs import TIME_STEP, PACKET_SIZE

def average_delays(nodes, time_step=TIME_STEP):
	for node in nodes:
		# assuming arrival at 50% means half a task arrives per mili second
		arrival_per_timestep = sum(node._arrivals_on_slices)/time_step * 0.001
		_task_delays = []
		for task_type in node._task_type_on_slices:
			_task_delays.append(task_type[1]*PACKET_SIZE/1e6) # in mili-seconds
		print("[INFO]",node,"in each timestep has average processing delays of", round(sum(_task_delays)/len(_task_delays),2),"ms",
			"and",arrival_per_timestep,"tasks arriving per millisecond")

