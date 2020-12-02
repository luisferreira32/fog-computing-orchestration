#!/usr/bin/env python
""" Provides an event to set up the fog nodes transmitting flags """

# >>>>> imports
from .core import Event
from .stop_transmitting import Stop_transmitting

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Start_transmitting(Event):
	""" Start_transmitting is an event that sets the transmission flag up

	Attributes:
		node: Fog_node - the fog node which is transmitting
		arrival_time: float - when the task will arrive and the node will stop transmitting
		bw: int - the bandwidth used for this transmission
	"""
	def __init__(self, time, node, arrival_time, bw):
		"""
		Parameters:
			(super) time: float - the time in which the event will run
			node: Fog_node - the fog node which is transmitting
			arrival_time: float - when the task will arrive and the node will stop transmitting
			bw: int - the bandwidth used for this transmission
		"""
		
		super(Start_transmitting, self).__init__(time, "Start_transmitting")
		self.node = node
		self.arrival_time = arrival_time
		self.bw = bw

	def execute(self,evq):
		""" Executes the flag change."""

		self.node.start_transmitting(self.bw)
		evq.add_event(Stop_transmitting(self.arrival_time, self.node, self.bw))
		return None

# <<<<<