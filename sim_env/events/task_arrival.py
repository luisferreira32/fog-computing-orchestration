#!/usr/bin/env python
"""Provides a class and auxiliary functions for the event responsible for making a task arrive at a certain node and slice"""

# >>>>> imports
from .core import Event
from utils.custom_exceptions import InvalidValueError

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class and functions
class Task_arrival(Event):
	""" Task_arrival is an event to either insert the task to a node slice or return it in case of overflow.

	Attributes:
		(super) time: float - the current simulation time
		node: Fog_nodes - an insntance of the class that represents a Fog node hardware strucutre
		k: int - the slice in which the task is arriving
		task: Task - the task that is arriving at the said slice and node
	"""
	def __init__(self, time, node, k, task):
		"""
		Parameters:
			time: float - the current simulation time
			node: Fog_nodes - an insntance of the class that represents a Fog node hardware strucutre
			k: int - the slice in which the task is arriving
			task: Task - the task that is arriving at the said slice and node
		"""

		super(Task_arrival, self).__init__(time, "Task_arrival")
		self.node = node
		self.k = k
		self.task = task
		# cannot recieve wrong slice
		if k >= node.max_k or k < 0:
			raise InvalidValueError("Invalid slice number", "[0,"+str(node.max_k)+"[")
		# cannot recieve future tasks
		if time < task.task_time():
			raise InvalidValueError("Cannot recieve a task from the future")

	def execute(self, evq):
		""" Executes a task arrival on a said node and slice, returning the task value if there was an overflow.

		Parameters:
			evq: Event_queue - the event queue from which this event was called and to which it can add events
		"""
		return self.node.add_task_on_slice(self.k, self.task)


def is_arrival_on_slice(ev, node, k):
	""" Small function to test if an event is an arrival on the node n and slice k

	Parameters:
		ev: Event - an event, subclass of Event(ABC)
		node: Fog_node - a fog node class instanciation
		k: int - a valid slice index/number
	"""
	return (ev.classtype == "Task_arrival" and ev.node == node and ev.k == k)

def is_offload_arrival_event(ev):
	""" Small function to test if an event is an arrival from an offloading event

	Parameters:
		ev: Event - an event, subclass of Event(ABC)
	"""
	return (ev.classtype == "Task_arrival" and ev.task.task_time() < ev.time)

# <<<<<