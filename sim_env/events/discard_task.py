#!/usr/bin/env python

from .core import Event

class Discard_task(Event):
	"""Discard_task that has its' delay constraint unmet
	"""
	def __init__(self, time, node, k, task):
		super(Discard_task, self).__init__(time, "Discard_task")
		self.node = node
		self.k = k
		self.task = task

	def execute(self, evq):
		return self.node.remove_task_of_slice(self.k, self.task)