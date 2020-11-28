#!/usr/bin/env python

# sim_env imports
from sim_env.configs import CPU_UNIT, RAM_UNIT, PACKET_SIZE
from sim_env.configs import DEBUG


def task_processing_time(t=None):
	# calculates the time a task takes to process after starting to process
	if t is None: return "no task given"
	if not t.is_processing(): return "task not processing"
	# task has cpu_demand cycles/bit
	total_cycles = t.cpu_demand*t.packet_size
	# processing units are in 1GHz each
	total_time = total_cycles/(t._cpu_units*CPU_UNIT*(1e9))
	return total_time

def task_communication_time(t, bit_rate):
	if t is None: return "no task given"
	if bit_rate == 0: return "invalid transmission route"
	# simple packet_size calc
	return (t.packet_size)/bit_rate

		
class Task(object):
	""" A task attributed by the users
	"""
	def __init__(self, timestamp, packet_size=PACKET_SIZE, delay_constraint=10, cpu_demand=400, ram_demand=400, task_type=None):
		super(Task, self).__init__()
		# must either have task type or the other
		self.packet_size = packet_size
		if not task_type == None and len(task_type) == 3:
			self.delay_constraint = task_type[0]
			self.cpu_demand = task_type[1]
			self.ram_demand = task_type[2]
		else:
			self.delay_constraint = delay_constraint
			self.cpu_demand = cpu_demand
			self.ram_demand = ram_demand

		self._processing = False
		self._memory_units = 0
		self._cpu_units = 0
		self._timestamp = timestamp
		self._total_delay = -1
		self._expected_delay = -1

	def __str__(self):
		return str(self._timestamp)+"s is_processing "+str(self._processing)

	def is_processing(self):
		return self._processing

	def is_completed(self):
		return False if self._total_delay == -1 else True

	def start_processing(self, cpu_units, memory_units):
		self._processing = True
		self._cpu_units = cpu_units
		self._memory_units = memory_units
		self._expected_delay = task_processing_time(self)

	def finish_processing(self, finish_time):
		self._processing = False
		self._total_delay = finish_time-self._timestamp

	def task_delay(self):
		return self._total_delay

	def task_time(self):
		return self._timestamp

	def constraint_time(self):
		return self._timestamp+0.001*self.delay_constraint

	def exceeded_contraint(self, current_time):
		return self.constraint_time() < current_time

