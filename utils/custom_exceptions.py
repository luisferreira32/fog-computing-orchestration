#!/usr/bin/env python

class InvalidValueError(Exception):
	"""InvalidValueError raises an exception for an invalid value
	"""
	def __init__(self, message, rg=None):
		super(InvalidValueError, self).__init__(message)
		self.msg = message
		self.rg = rg

	def __str__(self):
		if self.rg is None:
			self.rg = "[not defined here, verify documentation]"
		return f'{self.msg} -> acceptable range {self.rg}'
		