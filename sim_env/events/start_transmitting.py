#!/usr/bin/env python
""" Provides an event to set up the fog nodes transmitting flags """

# >>>>> imports
from .core import Event

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Start_transmitting(Event):
	""" Start_transmitting is an event that sets the transmission flag up

	Attributes:
		node: Fog_node - the fog node which is transmitting
	"""
	def __init__(self, time, node):
		"""
		Parameters:
			(super) time: float - the time in which the event will run
			node: Fog_node - the fog node which is transmitting
		"""
		
		super(Start_transmitting, self).__init__(time, "Start_transmitting")
		self.node = node

	def execute(self,evq):
		""" Executes the flag change."""

		self.node.start_transmitting()
		return None

# <<<<<