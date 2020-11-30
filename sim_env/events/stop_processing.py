#!/usr/bin/env python

from .core import Event

class Stop_processing(Event):
	""" Stop_processing specifies a task finished processing and returns it if it is completed
	"""
	def __init__(self, time, node, k, task):
		super(Stop_processing, self).__init__(time, "Stop_processing")
		self.node = node
		self.k = k
		self.task = task

	def execute(self, evq):
		self.task.stop_processing(self.time)
		return self.node.remove_task_of_slice(self.k, self.task) if self.task.is_completed() else None