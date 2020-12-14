#!/usr/bin/env python
""" This file contains a fog Task object and associated functions

The Task object has a set of attributes that will allow control over the resources
demands and time keeping, and a set of methods for changing and delivering those
attributes. The two auxiliary functions are a calculation of the task processing time
and a calculation of the task communication time, given the right parameters.
"""

# >>>>> imports
# sim_env imports
from sim_env import configs as cfg
# utils imports
from utils.custom_exceptions import InvalidValueError, InvalidStateError

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> functions and classes
def task_processing_time(t, cpu_units=1):
	""" Calculates the processing time of a task t, an instance of class Task.

	Parameters:
		t: Task - a task that started processing in the fog node

	Exceptions:
		InvalidStateError - raised when the task is not being processed
	"""
	# if it is processing it might have different cpu units attributed
	if t.is_processing():
		cpu_units = t._cpu_units
	# task has cpu_demand cycles/bit
	total_cycles = t.cpu_demand*t.packet_size
	# processing units are in 1GHz each
	total_time = total_cycles/(cpu_units*cfg.CPU_UNIT)
	return total_time

def task_communication_time(packet_size_, bit_rate):
	""" Calculates the transmission/communication time of a task t, an instance of class Task, given a bit_rate

	Parameters:
		packet_size_: int - a task packet size in bits
		bit_rate: float - the number of bits per second transmitted

	Exceptions:
		InvalidValueError - raised when the argument values are incorrect
	"""
	if bit_rate <= 0 or packet_size_ <= 0:
		raise InvalidValueError("task_communication_time must have a valid task with a positive bitrate")
	# simple packet_size calc
	return float(float(packet_size_)/bit_rate)

		
class Task(object):
	""" A task attributed by the users to a fog node

	Atributes:
		_timestamp: float - task creation timestamp in seconds
		packet_size: int - task packet size in bits
		delay_constraint: int - task delay constraint in milliseconds
		cpu_demand: int - task cpu density demand in cycles/bit
		ram_demand: int - task RAM demand in MB
		_processing: bool - if the task is currently in a fog node processor
		_memory_units: int - the number of memory units from the fog node attributed to this task
		_cpu_units: int - the number of cpu units from the fog node attributed to this task
		_total_delay: float - the total delay from creation to completion
		_started_processing: float - the timestamp where it started processing last
		_expected_delay: float - expected processing delay

	"""
	def __init__(self, timestamp, packet_size=cfg.PACKET_SIZE, delay_constraint=10, cpu_demand=400, ram_demand=400, task_type=None):
		"""
		Parameters:
			timestamp - the timestamp of task creation
			packet_size = PACKET_SIZE - packet size in bits of a task
			delay_constraint: int = 10 - the delay constraint of a task in milliseconds
			cpu_demand: int = 400 - the computing demand in cycles/bit
			ram_demand: int = 400 - the ram demand of a task in MB
			task_type = None - the three aforementioned attributes but in a list

		Exceptions:
			InvalidValueError - raised when any of the arguments is negative
		"""
		super(Task, self).__init__()
		# check arguments
		if delay_constraint < 0 or cpu_demand < 0 or ram_demand < 0 or packet_size < 0 or timestamp < 0:
			raise InvalidValueError("No arguments on Task object can be negative")
		self._timestamp = timestamp
		# must either have task type or the other
		self.packet_size = packet_size
		if not task_type == None and len(task_type) == 3: # BUG: task_type can be "hacked"
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
		self._total_delay = -1
		# keep track of delay
		self._started_processing = -1
		self._expected_delay = -1

	def __str__(self):
		return "[task:"+str(self._timestamp)+"s] is_processing "+str(self._processing)

	def is_processing(self):
		"""Returns if it is processing """
		return self._processing

	def is_completed(self):
		"""Returns if it is completed """
		return False if self._total_delay == -1 else True

	def start_processing(self, cpu_units, memory_units, start_time):
		"""Sets up the variables indicating that it has started processing
		
		Parameters:
			cpu_units: int - the number of cpu units from the fog node attributed to this task
			memory_units: int - the number of memory units from the fog node attributed to this task
			start_time: float - the starting time of the processing
		"""
		if cpu_units < 0 or memory_units < 0 or start_time < self._timestamp:
			raise InvalidValueError("Task start_processing arguments do not meet requirements")
		self._processing = True
		self._cpu_units = cpu_units
		self._memory_units = memory_units
		self._started_processing = start_time
		if self._expected_delay == -1: # only set the expected delay the first time
			self._expected_delay = task_processing_time(self)

	def stop_processing(self, finish_time):
		"""Stops task processing and verifies if the task has completed its processing
		
		Parameters:
			finish_time: float - the precise time when it finished processing in the fog node
		"""

		if finish_time < self._timestamp:
			raise InvalidValueError("Task cannot stop before creation")
		if self._processing:
			self._processing = False
			self._cpu_units = 0
			self._memory_units = 0
			# if it finished the whole processing
			if round(finish_time-self._started_processing,5) == round(self._expected_delay, 5):
				self._total_delay = finish_time-self._timestamp
				self._expected_delay = 0
			# else just keep track of new expected delay
			else:
				self._expected_delay -= finish_time-self._started_processing

	def task_remaining_processing_time(self):
		"""Returns the reamining processing time """

		return self._expected_delay

	def task_delay(self):
		"""Returns the total delay """

		return self._total_delay

	def task_time(self):
		"""Returns the task creation timestamp """

		return self._timestamp

	def constraint_time(self):
		"""Returns the task limit completion time"""

		return self._timestamp+0.001*self.delay_constraint

	def exceeded_constraint(self, current_time):
		"""Returns if it has exceeded the time constraint """

		return self.constraint_time() < current_time


# <<<<<