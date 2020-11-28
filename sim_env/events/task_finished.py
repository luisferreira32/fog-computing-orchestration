#!/usr/bin/env python

from .core import Event

class Task_finished(Event):
	""" Task_finished specifies a task finished processing and returns it
	"""
	def __init__(self, time, node, k, task):
		super(Task_finished, self).__init__(time, "Task_finished")
		self.node = node
		self.k = k
		self.task = task

	def execute(self, evq):
		self.task.finish_processing(self.time)
		self.node._dealt_tasks += 1
		return self.node.remove_task_of_slice(self.k, self.task)