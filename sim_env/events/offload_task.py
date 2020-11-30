#!/usr/bin/env python
"""Provides a class for the event of offloading a task to another node within a logical slice"""

# >>>>> imports
from sim_env.fog import Fog_node
from utils.custom_exceptions import InvalidValueError
from .core import Event
from .task_arrival import Task_arrival

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Offload_task(Event):
	""" Offloads the task that just arrived from a client.
	
	Attributes:
		node: Fog_node - the fog node in which the task is arriving
		k: int - the slice in which the task is arriving
		destination: Fog_node - the node to which the task is going to be offloaded
		arrival_time: float - the time it takes to offload from node to destination
	"""

	def __init__(self, time, node, k, destination, arrival_time):
		"""
		Parameters:
			(super) time: float - the time of the event execution
			node: Fog_node - the fog node in which the task is arriving
			k: int - the slice in which the task is arriving
			destination: Fog_node - the node to which the task is going to be offloaded
			arrival_time: float - the time it takes to offload from node to destination
		"""

		super(Offload_task, self).__init__(time, "Offload_task")
		self.node = node
		self.k = k
		self.destination = destination
		self.arrival_time = arrival_time

		if k >= node.max_k or k < 0 or not isinstance(node, Fog_node) or not isinstance(destination, Fog_node):
			raise InvalidValueError("Verify arguments of Discard_task creation")

	def execute(self, evq):
		""" Executes the offloading event, not offloading if node is transmitting or there are no valid tasks to offload.

		Parameters:
			evq: Event_queue - the event queue from which this event was called and to which it can add events
		"""

		# can't send if it's busy sending
		if self.node.is_transmitting(): return None
		# then pop the last task we got
		t = self.node.pop_task_to_send(self.k, self.time)
		# if it's an invalid choice return without sending out the task
		if t == None: return None
		# else plan the landing
		evq.add_event(Task_arrival(self.arrival_time, self.destination, self.k, t))
		return None


