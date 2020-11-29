#!/usr/bin/env python

from .core import Event

class Finished_transmitting(Event):
	""" Finished_transmitting is an event that sets the transmission flag down
	"""
	def __init__(self, time, node):
		super(Finished_transmitting, self).__init__(time, "Finished_transmitting")
		self.node = node

	def execute(self,evq):
		self.node.finished_transmitting()
		return None
