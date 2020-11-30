#!/usr/bin/env python
"""Provides a class for the event of stopping to process a task in a certain node and slice, if the task has finished the processing it's returned"""

# >>>>> imports
from sim_env.fog import Task, Fog_node
from utils.custom_exceptions import InvalidValueError

from .core import Event

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Stop_processing(Event):
	""" Stop_processing specifies a task stops processing and returns it if it is completed
	
	Attributes:
		node: Fog_node - the fog node in which the task is processing
		k: int - the slice in which the task is processing
		task: Task - the task that will be stopped
	"""

	def __init__(self, time, node, k, task):
		"""
		Parameters:
			(super) time: float - the time in which the processing will stop
			node: Fog_node - the fog node in which the task is processing
			k: int - the slice in which the task is processing
			task: Task - the task that will be stopped
		"""

		super(Stop_processing, self).__init__(time, "Stop_processing")
		self.node = node
		self.k = k
		self.task = task

		if not isinstance(node, Fog_node) or k >= node.max_k or k < 0 or not isinstance(task, Task):
			raise InvalidValueError("Verify arguments of Stop_processing creation")

	def execute(self, evq):
		""" Executes a stop processing event on slice k of the node, stopping the task and returning it if completed.

		Parameters:
			evq: Event_queue - the event queue from which this event was called and to which it can add events
		"""

		self.node.stop_processing_in_slice(self.k, self.task, self.time)
		return self.node.remove_task_of_slice(self.k, self.task) if self.task.is_completed() else None

# <<<<<