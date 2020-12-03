#!/usr/bin/env python
"""Provides a class for the event of offloading a task to another node within a logical slice"""

# >>>>> imports
from sim_env.fog import Fog_node, task_communication_time, point_to_point_transmission_rate
from sim_env.configs import NODE_BANDWIDTH_UNIT
from utils.custom_exceptions import InvalidValueError
from .core import Event
from .task_arrival import Task_arrival
from .start_transmitting import Start_transmitting

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
		concurr: int - number of concurrent offloads happening in this node
	"""

	def __init__(self, time, node, k, destination, concurr):
		"""
		Parameters:
			(super) time: float - the time of the event execution
			node: Fog_node - the fog node in which the task is arriving
			k: int - the slice in which the task is arriving
			destination: Fog_node - the node to which the task is going to be offloaded
			concurr: int - number of concurrent offloads happening in this node
		"""

		super(Offload_task, self).__init__(time, "Offload_task")
		self.node = node
		self.k = k
		self.destination = destination
		self.concurr = concurr

		if not isinstance(node, Fog_node) or k >= node.max_k or k < 0 or not isinstance(destination, Fog_node) or concurr < 1:
			raise InvalidValueError("Verify arguments of Discard_task creation")

	def execute(self, evq):
		""" Executes the offloading event, not offloading if node is transmitting or there are no valid tasks to offload.

		Parameters:
			evq: Event_queue - the event queue from which this event was called and to which it can add events
		"""

		# can't send if it's busy sending - and doesn't have a unit available for all transmissions
		bw = int(self.node.available_bandwidth()/self.concurr)
		if bw < NODE_BANDWIDTH_UNIT:
			return None
		# then pop the last task we got
		t = self.node.pop_task_to_send(self.k, self.time)
		# if it's an invalid choice return without sending out the task
		if t == None:
			return None
		# else plan the landing
		arrival_time = self.time+ task_communication_time(t.packet_size, point_to_point_transmission_rate(self.node._distances[self.destination.index], bw))
		evq.add_event(Task_arrival(arrival_time, self.destination, self.k, t))
		evq.add_event(Start_transmitting(self.time, self.node, arrival_time, bw))
		return None


