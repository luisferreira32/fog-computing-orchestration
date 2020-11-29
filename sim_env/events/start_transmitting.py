#!/usr/bin/env python

from .core import Event

class Start_transmitting(Event):
	""" Start_transmitting is an event that sets the transmission flag up
	"""
	def __init__(self, time, node):
		super(Start_transmitting, self).__init__(time, "Start_transmitting")
		self.node = node

	def execute(self,evq):
		self.node.start_transmitting()
		return None