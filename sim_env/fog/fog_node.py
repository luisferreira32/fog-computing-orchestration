#!/usr/bin/env python

from .core import Node
import numpy as np

# sim_env imports
from sim_env.configs import MAX_QUEUE, CPU_UNIT, RAM_UNIT, CPU_CLOCKS, RAM_SIZES, BASE_SLICE_CHARS, DEFAULT_SLICES
from sim_env.configs import AREA, PATH_LOSS_CONSTANT, PATH_LOSS_EXPONENT, THERMAL_NOISE_DENSITY
from sim_env.configs import NODE_BANDWIDTH, TRANSMISSION_POWER, PACKET_SIZE
from sim_env.configs import DEBUG
from sim_env.fog import euclidean_distance, channel_gain, shannon_hartley, db_to_linear

from utils.tools import uniform_rand_choice, uniform_rand_int


def point_to_point_transmission_rate(n1, n2):
	# calculates transmission rate given two nodes
	d = euclidean_distance(n1.x, n1.y, n2.x, n2.y)
	g = channel_gain(d, PATH_LOSS_CONSTANT, PATH_LOSS_EXPONENT)
	p_mw = db_to_linear(TRANSMISSION_POWER)
	n0_mw = db_to_linear(THERMAL_NOISE_DENSITY)
	return shannon_hartley(g, p_mw, NODE_BANDWIDTH, n0_mw)

def create_random_node(index=0, slices_characteristics=BASE_SLICE_CHARS):
	# returns a node uniformly sampled within configurations, after the previous index
	[x, y] = [uniform_rand_int(low=0, high=AREA[0]), uniform_rand_int(low=0, high=AREA[1])]
	number_of_slices = DEFAULT_SLICES
	cpu = uniform_rand_choice(CPU_CLOCKS)
	ram = uniform_rand_choice(RAM_SIZES)
	if DEBUG: print("[DEBUG] Node",index,"created at (x,y) = ",(x,y),"cpu =",cpu,"ram =",ram)
	return Fog_node(index, x, y, cpu, ram, number_of_slices, slices_characteristics)


class Fog_node(Node):
	""" A Fog node with limited resources
	"""
	def __init__(self, index, x, y, cpu_frequency, ram_size, number_of_slices, slice_characteristics=BASE_SLICE_CHARS):
		super(Fog_node, self).__init__(index, x, y, cpu_frequency, ram_size, number_of_slices)
		# service rate constants
		self._dealt_tasks = 0
		self._total_time_intervals = 0
		self._service_rate = 0
		# task slices constants
		self._arrivals_on_slices = [ar for ar in slice_characteristics["arrivals"]]
		self._task_type_on_slices = [tp for tp in slice_characteristics["task_type"]]
		# com times within fog nodes
		self._communication_rates = {}
		self._distances = {}
		# keep track of processed tasks
		self._being_processed = np.zeros(number_of_slices, dtype=np.uint8)

	def set_communication_rates(self, nodes):
		for n in nodes:
			self._distances[n.index]= euclidean_distance(self.x, self.y, n.x, n.y)
			self._communication_rates[n.index] = point_to_point_transmission_rate(self,n)
		if DEBUG: print("[DEBUG]",self.name, [str(round(r/1000000,2))+" Kb/ms" for _,r in self._communication_rates.items()])

	def new_interval_update_service_rate(self):
		self._total_time_intervals += 1
		self._service_rate = self._dealt_tasks / self._total_time_intervals
		# to avoid zero divisions
		if self._service_rate == 0:
			self._service_rate = 1

	def slice_buffer_len(self, k):
		# error shield
		if not (0<=k<self.max_k): return 0
		return len(self.buffers[k])

	# override
	def add_task_on_slice(self, k, task):
		# error shield
		if not (0<=k<self.max_k): return task
		# returns the task if the slice buffer is full
		if len(self.buffers[k]) == self.buffers[k].maxlen:
		    return task
		self.buffers[k].append(task)
		return None
    
    # override
	def pop_last_task(self, k, time):
		# error shield
		if not (0<=k<self.max_k): return None
		# only works if there is a task in the buffer
		if len(self.buffers[k]) == 0: return None
		# shouldn't pop a task that is processing
		if self.buffers[k][-1].is_processing(): return None
		# and should only offload a task arriving in this timestep
		if self.buffers[k][-1]._timestamp != time: return None
		return self.buffers[k].pop()

	# override
	def remove_task_of_slice(self, k, task):
		# error shield
		if not (0<=k<self.max_k): return None
		# removes and returns a task if it is in the buffer
		try:
			self.buffers[k].remove(task)
			if task.is_completed() or task.is_processing(): self._being_processed[k] -= 1
			# values should be zero if it's not processing
			self._avail_ram_units += task._memory_units
			self._avail_cpu_units += task._cpu_units
		except:
			return None
		return task

	# override
	def start_processing_in_slice(self, k, w):
		under_processing = [];
		# error shield
		# if w <= 0 or not (0<=k<self.max_k): return under_processing
		# only process if there is a task on the buffer
		for task in self.buffers[k]:
			# only process if has cores and memory for it
			if not task.is_processing() and w > 0 and self._avail_cpu_units > 0 and self._avail_ram_units >= np.ceil(task.ram_demand/RAM_UNIT):
				# one cpu unit per task and the ram demand they require
				n_cpu_units = 1
				n_memory_units = np.ceil(task.ram_demand/RAM_UNIT)
				# and take them from the available pool
				self._avail_cpu_units -= n_cpu_units
				self._avail_ram_units -= n_memory_units
				# then start the processing
				task.start_processing(n_cpu_units, n_memory_units)
				under_processing.append(task)
				# reduce the number that we will still allocate
				w -= 1
				self._being_processed[k] += 1
		return under_processing

	# override
	def reset(self):
		for i in range(self.max_k):
			self.buffers[i].clear()
		self._avail_cpu_units = int(self.cpu_frequency/CPU_UNIT)
		self._avail_ram_units = int(self.ram_size/RAM_UNIT)
