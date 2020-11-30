#!/usr/bin/env python
"""Provides a class for the event of discarding a task due to a time constraint being exceeded"""

# >>>>> imports
from sim_env.fog import Task, Fog_node
from utils.custom_exceptions import InvalidValueError

from .core import Event

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Discard_task(Event):
	"""Discard_task that has its delay constraint unmet
	
	Attributes:
		node: Fog_node - the fog node in which the task is in the buffer
		k: int - the slice in which the task currently is placed
		task: Task - the task that had its delay constraint unmet
	"""
	def __init__(self, time, node, k, task):
		"""
		Parameters:
			(super) time: float - the time of the event execution
			node: Fog_node - the fog node in which the task is in the buffer
			k: int - the slice in which the task currently is placed
			task: Task - the task that had its delay constraint unmet
		"""

		super(Discard_task, self).__init__(time, "Discard_task")
		self.node = node
		self.k = k
		self.task = task

		if not isinstance(node, Fog_node) or k >= node.max_k or k < 0 or not isinstance(task, Task):
			raise InvalidValueError("Verify arguments of Discard_task creation")

	def execute(self, evq):
		""" Executes the discarding event, restoring allocated resources back to the node. If the task was already not in the node the execute returns None.

		Parameters:
			evq: Event_queue - the event queue from which this event was called and to which it can add events
		"""
		
		return self.node.remove_task_of_slice(self.k, self.task)

# <<<<<