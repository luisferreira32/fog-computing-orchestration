#!/usr/bin/env python
"""Custom exceptions and errors defenition"""
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> classes
class InvalidValueError(Exception):
	"""InvalidValueError raises an exception for an invalid value
	"""
	def __init__(self, message, rg="[not defined here, verify documentation]"):
		super(InvalidValueError, self).__init__(message)
		self.msg = message
		self.rg = rg

	def __str__(self):
		return f'{self.msg} -> acceptable range {self.rg}'

class InvalidStateError(Exception):
	"""InvalidStateError raises an exception for an invalid state of an object
	"""
	def __init__(self, message, st="[unkown]"):
		super(InvalidStateError, self).__init__(message)
		self.msg = message
		self.st = st

	def __str__(self):
		return f'{self.msg} -> object state {self.st}'
		
# <<<<<
		