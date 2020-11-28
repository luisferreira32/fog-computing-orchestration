#!/usr/bin/env python

from .core import Event

from sim_env.fog import Task, task_processing_time, task_communication_time

class Offload_task(Event):
	""" Offloads the task that just arrived to a destination node
	"""
	def __init__(self, time, node, k, destination, con=1):
		super(Offload, self).__init__(time, "Offload")
		self.node = node
		self.k = k
		self.destination = destination
		self.concurrent_offloads = con

	def execute(self, evq):
		# can't send if there is no way to send or it's busy sending
		if self.node._communication_rates[self.destination.index] == 0: return None
		if self.node.transmitting: return None
		# then pop the last task we got
		t = self.node.pop_last_task(self.k, self.time)
		# if it's an invalid choice return without sending out the task
		if t == None: return None
		# else plan the landing
		self.node._dealt_tasks += 1
		self.node.transmitting = True
		arrive_time = self.time + task_communication_time(t, self.node._communication_rates[self.destination.index]/self.concurrent_offloads)
		evq.add_event(Task_arrival(arrive_time, self.destination, self.k, t))
		evq.add_event(Finished_transmitting(arrive_time, self.node))
		return None


