#!/usr/bin/env python
"""Provides a class and auxiliary functions for the event responsible for setting arrivals according to a bernoulli distribution."""

# >>>>> imports
from .core import Event
from .task_arrival import Task_arrival

from sim_env.fog import bernoulli_arrival, Task
# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Set_arrivals(Event):
	""" Set_arrivals calculates which nodes and slices are recieving a task this timestep.

	Attributes:
		(super) time: float - the current simulation time
		timestep: float - the timestep taken until the next arrival calculation
		nodes: List[Fog_nodes] - a list of fog nodes to which tasks will arrive at their buffers
		case: case_struct - a structure defined in sim_env.configs settling slices characteristics 
	"""

	def __init__(self, time, timestep, nodes, case):
		"""
		Parameters:
			time: float - the current simulation time
			timestep: float - the timestep taken until the next arrival calculation
			nodes: List[Fog_nodes] - a list of fog nodes to which tasks will arrive at their buffers
			case: case_struct - a structure defined in sim_env.configs settling slices characteristics 
		"""

		super(Set_arrivals, self).__init__(time, "Set_arrivals")
		self.timestep = timestep
		self.nodes = nodes
		self.case = case

	def execute(self, evq):
		""" Executes it's function, setting arrivals for the fog slices based on the case task_type and arrival rate.

		Each slice on a case has a different task_type characterized by resources and constraints, to create a Task object.
		This task object has chance of arrival modelled by the bernoulli distribution with p decided per slice on the case.
		Although every event will only run on the next timestep, note that every Task_arrival event will run before
		the next timestep Set_arrivals.

		Parameters:
			evq: Event_queue - the event queue from which this event was called and to which he will add events
		"""
		
		# for each node slice, check wether the task arrived or not, and place an event for the next timestep
		for n in self.nodes:
			for i in range(n.max_k):
				if bernoulli_arrival(self.case["arrivals"][i]):
					t = Task(self.time+self.timestep, task_type=self.case["task_type"][i])
					ev = Task_arrival(self.time+self.timestep, n, i, t)
					evq.add_event(ev)
		# then recursevly ask for another set of arrivals
		evq.add_event(Set_arrivals(self.time+self.timestep, self.timestep, self.nodes, self.case))
		return None

# <<<<<