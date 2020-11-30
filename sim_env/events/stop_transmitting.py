#!/usr/bin/env python
""" Provides an event to set up the fog nodes transmitting flags """

# >>>>> imports
from .core import Event

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> class
class Stop_transmitting(Event):
	""" Stop_transmitting is an event that sets the transmission flag down

	Attributes:
		node: Fog_node - the fog node which is transmitting
	"""

	def __init__(self, time, node):
		"""
		Parameters:
			(super) time: float - the time in which the event will run
			node: Fog_node - the fog node which is transmitting
		"""

		super(Stop_transmitting, self).__init__(time, "Stop_transmitting")
		self.node = node

	def execute(self,evq):
		""" Executes the flag change."""

		self.node.finished_transmitting()
		return None

# <<<<<