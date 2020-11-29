#!/usr/bin/env python

from .core import Event
from .discard_task import Discard_task
from .task_finished import Task_finished

from sim_env.fog import task_processing_time

class Start_processing(Event):
	""" Start_processing executes the task of starting to process w tasks
	"""
	def __init__(self, time, node, k, w):
		super(Start_processing, self).__init__(time, "Start_processing")
		self.node = node
		self.k = k
		self.w = w
		
	def execute(self, evq):
		tasks_under_processing, discarded = self.node.start_processing_in_slice(self.k, self.w, self.time)
		# discard and set finish processing when decisions are made
		for task in tasks_under_processing:
			finish = self.time+task_processing_time(task)
			if task.exceeded_contraint(finish):
				evq.add_event(Discard_task(max(task.constraint_time(), self.time), self.node, self.k, task))
			else:
				evq.add_event(Task_finished(finish, self.node, self.k, task))
		for task in discarded:
			evq.add_event(Discard_task(max(task.constraint_time(), self.time), self.node, self.k, task))
							
		return None
