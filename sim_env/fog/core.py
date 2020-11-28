#!/usr/bin/env python

# external imports
from abc import ABC, abstractmethod
from collections import deque

from sim_env.configs import MAX_QUEUE, CPU_UNIT, RAM_UNIT

class Node(ABC):
	""" Abstract node class
	"""
	def __init__(self, index, x, y, cpu_frequency, ram_size, number_of_slices):
		super().__init__()
		# Identifiers
		self.index = index
		self.name = "node_"+str(index)
		# placement
		self.x = x
		self.y = y
		# resources
		self.cpu_frequency = cpu_frequency
		self.ram_size = ram_size
		self._avail_cpu_units = cpu_frequency/CPU_UNIT
		self._avail_ram_units = int(ram_size/RAM_UNIT)
		# slices buffers
		self.max_k = number_of_slices
		self.buffers = [deque(maxlen=MAX_QUEUE) for _ in range(number_of_slices)]
		# states
		self.transmitting = False

	def __str__(self):
		return self.name

	@abstractmethod
	def add_task_on_slice(self, k, task):
		# returns the task if the slice buffer is full
		pass

	@abstractmethod
	def remove_task_of_slice(self, k, task):
		# removes and returns a task if it is or not on the buffer
		pass

	@abstractmethod
	def start_processing_in_slice(self, k, w):
		# starts processing w tasks in slice k
		pass

	@abstractmethod
	def reset(self):
		# resets the state of the node
		pass