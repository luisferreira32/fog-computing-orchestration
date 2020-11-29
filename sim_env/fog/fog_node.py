#!/usr/bin/env python
"""Provides a class and auxiliary functions to model a Fog computing node hardware.

Fog_node class has the essential characteristics of a Fog node and auxiliary variables
for the simulations, provides methods for observation, buffer and cpu queueing, and for
starting and stopping a task transmission. Two auxiliary functions are also implemented,
one to calculate transmission rate between two Fog_nodes, and the other to create a random
node based on the sim_env.configs constants.
"""

# >>>>> imports
import numpy as np
from collections import deque

# sim_env imports
from sim_env.configs import MAX_QUEUE, CPU_UNIT, RAM_UNIT, CPU_CLOCKS, RAM_SIZES, BASE_SLICE_CHARS, DEFAULT_SLICES
from sim_env.configs import AREA, PATH_LOSS_CONSTANT, PATH_LOSS_EXPONENT, THERMAL_NOISE_DENSITY
from sim_env.configs import NODE_BANDWIDTH, TRANSMISSION_POWER, PACKET_SIZE
from sim_env.configs import DEBUG
from .calculators import euclidean_distance, channel_gain, shannon_hartley, db_to_linear
from .task import Task

# utils
from utils.tools import uniform_rand_choice, uniform_rand_int
from utils.custom_exceptions import InvalidValueError

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> classes and functions
def point_to_point_transmission_rate(d, concurr):
	""" Calculates the transmission rate over a noisy channel between two nodes

	Parameters:
		d: float - distance between two nodes
		concurr: int - number of tasks that will share the bandwidth
	"""
	if concurr <= 0:
		raise InvalidValueError("transmission rate number of task has to be positive", "[0,+inf[")
	g = channel_gain(d, PATH_LOSS_CONSTANT, PATH_LOSS_EXPONENT)
	p_mw = db_to_linear(TRANSMISSION_POWER)
	n0_mw = db_to_linear(THERMAL_NOISE_DENSITY)
	return shannon_hartley(g, p_mw, NODE_BANDWIDTH/concurr, n0_mw)

def create_random_node(index=0):
	""" Creates an instance of the Fog_node class, based on the configurations in sim_env.configs

	Parameters:
		index: int - the node identifier (and index on a node list)
	"""
	[x, y] = [uniform_rand_int(low=0, high=AREA[0]), uniform_rand_int(low=0, high=AREA[1])]
	number_of_slices = DEFAULT_SLICES
	cpu = uniform_rand_choice(CPU_CLOCKS)
	ram = uniform_rand_choice(RAM_SIZES)
	if DEBUG:
		print("[DEBUG] Node",index,"created at (x,y) = ",(x,y),"cpu =",cpu,"ram =",ram)
	return Fog_node(index, x, y, cpu, ram, number_of_slices)


class Fog_node():
	""" A Fog node with limited resources and a defined number of SDN slices. Represents the hardware.

	It is assumed that the node is placed in a plane. Its communications are in a full duplex fashion,
	sharing its badwidth between all offloaded tasks in a timeslot.

	Attributes:
		index: int - identifier of the fog node in a list
		name: str - identifier of the fog node in a printing statement

		x: float - placement alongside the X-axis
		y: float - placement alongside the Y-axis
		_distances: disc{key: float} - distances between a group of Fog nodes instances

		cpu_frequency: int - cpu frequency in GHz
		ram_size: int - the size of the RAM in MB
		_avail_cpu_units: int - available CPU units (1 unit = 1 Gz)
		_avail_ram_units: int - available RAM units (1 unit = 400MB)

		max_k: int - number of slices
		buffers: List[deque] - a list of max_k deques that represents the buffers of each slice

		_dealt_tasks: int - the number of tasks this node dealt with
		_total_time_intervals: int - the number of time intervals this node survived
		_service_rate: float - service rate (task/interval) of this node

		_being_pocessed: np.array() - an array with size max_k, counting the number of tasks under processing in each buffer
		_transmitting: bool - a flag to indicate the node is transmitting

	"""

	def __init__(self, index, x, y, cpu_frequency, ram_size, number_of_slices):
		"""
		Parameters:
			index: int - identifier of the fog node in a list
			x: float - placement alongside the X-axis
			y: float - placement alongside the Y-axis
			cpu_frequency: int - cpu frequency in GHz
			ram_size: int - the size of the RAM in MB
			number_of_slices: int - number of software defined slices in the fog

		Exceptions:
			InvalidValueError - raised when any of the arguments is negative
		"""
		super(Fog_node, self).__init__()
		# check arguments
		if index<0 or x < 0 or y < 0 or cpu_frequency < 0 or ram_size < 0 or number_of_slices < 0:
			raise InvalidValueError("No arguments on Fog_node object can be negative")
		self.index = index
		self.name = "node_"+str(index)
		self.x = x
		self.y = y
		self._distances = {}
		self.cpu_frequency = cpu_frequency
		self.ram_size = ram_size
		self._avail_cpu_units = cpu_frequency/CPU_UNIT
		self._avail_ram_units = int(ram_size/RAM_UNIT)
		self.max_k = number_of_slices
		self.buffers = [deque(maxlen=MAX_QUEUE) for _ in range(number_of_slices)]
		self._dealt_tasks = 0
		self._total_time_intervals = 0
		self._service_rate = 0
		self._being_processed = np.zeros(number_of_slices, dtype=np.uint8)
		self._transmitting = False

	def __str__(self):
		return self.name

	# >>>>> slice management methods
	def slice_buffer_len(self, k):
		"""Retrieves the lenght of a buffer		

		Parameters:
			k: int - the slice index
		"""
		
		if k < 0 or k >= self.max_k:
			raise InvalidValueError("Invalid slice number", "[0,"+str(self.max_k)+"[")
		return len(self.buffers[k])

	def being_processed_on_slice(self, k):
		"""Retrieves the number of tasks being processed in the buffer

		Parameters:
			k: int - the slice index
		"""
		
		if k < 0 or k >= self.max_k:
			raise InvalidValueError("Invalid slice number", "[0,"+str(self.max_k)+"[")
		return self._being_processed[k]

	def add_task_on_slice(self, k, task):
		"""Tries to add a task on the buffer, if it overflows, returns the discarded task

		Parameters:
			k: int - the slice index
			task: Task - task to be added to buffer in slice k
		"""

		if k < 0 or k >= self.max_k or not isinstance(task, Task):
			raise InvalidValueError("Invalid arguments for add_task_on_slice")
		# returns the task if the slice buffer is full
		if len(self.buffers[k]) == self.buffers[k].maxlen:
		    return task
		self.buffers[k].append(task)
		return None
    
	def remove_task_of_slice(self, k, task):
		"""Try to remove a task from the buffer.
		
		Returns the task if it was indeed on this buffer, and gives back the memory and cpu
		units the task might have had. (the task only has them when it's processing)

		Parameters:
			k: int - the slice index
			task: Task - task to be removed from buffer in slice k
		"""

		if k < 0 or k >= self.max_k or not isinstance(task, Task):
			raise InvalidValueError("Invalid arguments for remove_task_of_slice")
		# removes and returns a task if it is in the buffer
		try:
			self.buffers[k].remove(task)
			if task.is_completed() or task.is_processing():
				self._being_processed[k] -= 1
			if task.is_completed():
				self._dealt_tasks += 1
			# values should be zero if it's not processing
			self._avail_ram_units += task._memory_units
			self._avail_cpu_units += task._cpu_units
		except:
			return None
		return task

	def start_processing_in_slice(self, k, w, time):
		"""Try to start processing w task, if any task has already exceeded time limit, discard it.

		On the slice k, starting at time, try to queue w tasks to processing. It depends
		on the bottleneck (cpu or ram) the amount that actually will start processing. If a task that
		would start processing has already exceeded its constraint, discard it instead.

		Parameters:
			k: int - the slice index
			w: int - number of tasks to attempt to process
			time: float - current simulation time
		"""

		if k < 0 or k >= self.max_k or w <=0 or time < 0:
			raise InvalidValueError("Invalid arguments for start_processing_in_slice")

		under_processing = []; discarded = [];
		# only process if there is a task on the buffer
		for task in self.buffers[k]:
			# only process if has cores, memory and an action request W for it
			if not task.is_processing() and w > 0 and self._avail_cpu_units > 0 and self._avail_ram_units >= np.ceil(task.ram_demand/RAM_UNIT):
				# if processor tries to load them and they exceeded constraint, move on
				if task.exceeded_contraint(time):
					discarded.append(task)
					continue
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
		return under_processing, discarded

	# <<<<<
	# >>>>> communication related methods
	def pop_task_to_send(self, k, time):
		"""Tries pop last recieved task in slice k to send. Only valid if recieved within this timestep
		
		Parameters:
			k: int - the slice index
			time: float - current simulation time
		"""

		if k < 0 or k >= self.max_k:
			raise InvalidValueError("Invalid slice number", "[0,"+str(self.max_k)+"[")
		# only works if there is a task in the buffer
		if len(self.buffers[k]) == 0: return None
		# shouldn't pop a task that is processing
		if self.buffers[k][-1].is_processing(): return None
		# and should only offload a task arriving in this timestep
		if self.buffers[k][-1]._timestamp != time: return None
		self._dealt_tasks += 1
		return self.buffers[k].pop()

	def finished_transmitting(self):
		"""Set transmitting flag to False"""

		self._transmitting = False

	def start_transmitting(self):
		"""Set transmitting flag to True"""

		self._transmitting = True

	def is_transmitting(self):
		"""Verify if the node is in the middle of a transmission"""

		return self._transmitting

	# <<<<<
	# >>>>> aux methods
	def set_distances(self, nodes):
		"""Given a list sets the _distances dict between instances of Fog_nodes class with an euclidean distance
		
		Parameters:
			nodes: List[Fog_nodes] - list of the neighbouring nodes of this node with DISTINCT indexes
		"""

		for n in nodes:
			if n.index == self.index: continue
			self._distances[n.index]= euclidean_distance(self.x, self.y, n.x, n.y)

	def new_interval_update_service_rate(self):
		"""Updates the service rate of this node every new interval """

		self._total_time_intervals += 1
		self._service_rate = self._dealt_tasks / self._total_time_intervals
		# to avoid zero divisions
		if self._service_rate == 0:
			self._service_rate = 1

	def reset(self):
		"""Resets the node state, clearing the buffers and restoring the resources """

		for i in range(self.max_k):
			self.buffers[i].clear()
		self._avail_cpu_units = int(self.cpu_frequency/CPU_UNIT)
		self._avail_ram_units = int(self.ram_size/RAM_UNIT)

	# <<<<<
# <<<<<