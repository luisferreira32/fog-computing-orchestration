#!/usr/bin/env python
"""Provides a class for the event of starting to process W tasks in a certain node and slice, if tasks do not meet time constraint it discards them."""

# >>>>> imports
import numpy as np

from sim_env.fog import Fog_node
from sim_env.configs import TIME_STEP
from utils.custom_exceptions import InvalidValueError

from .core import Event
from .discard_task import Discard_task
from .stop_processing import Stop_processing

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Start_processing(Event):
	""" Start_processing starts the first W tasks that can be processed in the slice k of the node.

	The processing is simulated by the starting of the process and the setting up of the event to stop its processing.
	Any event that modifies the task before its completion will "null" the Stop_processing event, and if the task exceeds
	its time constraints, the Start_processing event just sets a Discard_task event instead of Stop_processing event.

	Attributes:
		node: Fog_node - the fog node in which the tasks will process
		k: int - the slice in which the task is processing
		w: int - the number of tasks that will be attempted to start processing
	"""

	def __init__(self, time, node, k, w):
		"""
		Parameters:
			(super) time: float - the time in which the processing will start
			node: Fog_node - the fog node in which the tasks will process
			k: int - the slice in which the task is processing
			w: int - the number of tasks that will be attempted to start processing
		"""

		super(Start_processing, self).__init__(time, "Start_processing")
		self.node = node
		self.k = k
		self.w = w

		if not isinstance(node, Fog_node) or k >= node.max_k or k < 0 or w <= 0:
			raise InvalidValueError("Verify arguments of Start_processing creation")
		
	def execute(self, evq):
		""" Executes a start processing event in which W tasks will attempt to be processed, and any time constraint exceeded will result on a discard.

		Parameters:
			evq: Event_queue - the event queue from which this event was called and to which it can add events
		"""

		tasks_under_processing, discarded = self.node.start_processing_in_slice(self.k, self.w, self.time)
		# discard and set finish processing when decisions are made
		for task in tasks_under_processing:
			task_finish = self.time+task.task_remaining_processing_time()
			task_interruption = self.time+TIME_STEP
			if task.exceeded_constraint(min(task_finish, task_interruption)):
				evq.add_event(Discard_task(max(task.constraint_time(), self.time), self.node, self.k, task))
			else:
				# only process one timestep at a time
				evq.add_event(Stop_processing(min(task_finish, task_interruption), self.node, self.k, task))
		for task in discarded:
			evq.add_event(Discard_task(max(task.constraint_time(), self.time), self.node, self.k, task))
							
		return None
