#!/usr/bin/env python

from .core import Event
from .task_arrival import Task_arrival

from sim_env.fog import bernoulli_arrival, Task

class Set_arrivals(Event):
	""" Set_arrivals calculates which nodes and slices are recieving a task this timestep
	"""
	def __init__(self, time, timestep, nodes):
		super(Set_arrivals, self).__init__(time, "Set_arrivals")
		self.timestep = timestep
		self.nodes = nodes

	def execute(self, evq):
		# for each node slice, check wether the task arrived or not, and place an event for the next timestep
		for n in self.nodes:
			for i in range(n.max_k):
				if bernoulli_arrival(n._arrivals_on_slices[i]):
					t = Task(self.time+self.timestep, task_type=n._task_type_on_slices[i])
					ev = Task_arrival(self.time+self.timestep, n, i, t)
					evq.add_event(ev)
		# then recursevly ask for another set of arrivals
		evq.add_event(Set_arrivals(self.time+self.timestep, self.timestep, self.nodes))
		return None