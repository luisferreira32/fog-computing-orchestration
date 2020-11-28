#!/usr/bin/env python

from .core import Event

class Task_arrival(Event):
	""" Task_arrival inserts a task to a node slice, if it overflows returns the task
	"""
	def __init__(self, time, node, k, task):
		super(Task_arrival, self).__init__(time, "Task_arrival")
		self.node = node
		self.k = k
		self.task = task
		assert time >= task.task_time()

	def execute(self, evq):
		# if the task just arrived, schedule a discard for it's deadline (in milliseconds) # NOTE: only when deciding for now
		# evq.addEvent(Discard_task(self.task.task_time()+(0.001*self.task.delay_constraint), self.node, self.k, self.task))
		return self.node.add_task_on_slice(self.k, self.task)

	def task_time(self):
		return self.task.task_time()


def is_arrival_on_slice(ev, node, k):
	return (ev.classtype == "Task_arrival" and ev.node == node and ev.k == k)

def is_offload_arrival_event(ev, clock):
	return (ev.classtype == "Task_arrival" and ev.task_time() < clock)