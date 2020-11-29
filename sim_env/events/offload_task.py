#!/usr/bin/env python

from .core import Event
from .task_arrival import Task_arrival

class Offload_task(Event):
	""" Offloads the task that just arrived to a destination node
	"""
	def __init__(self, time, node, k, destination, arrival_time):
		super(Offload_task, self).__init__(time, "Offload_task")
		self.node = node
		self.k = k
		self.destination = destination
		self.arrival_time = arrival_time

	def execute(self, evq):
		# can't send if it's busy sending
		if self.node.is_transmitting(): return None
		# then pop the last task we got
		t = self.node.pop_task_to_send(self.k, self.time)
		# if it's an invalid choice return without sending out the task
		if t == None: return None
		# else plan the landing
		evq.add_event(Task_arrival(self.arrival_time, self.destination, self.k, t))
		return None


