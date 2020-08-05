# import necessary fog environment configurations
from . import configs

class Unit(object):
	"""
	A task unit to be processed in a fog CPU.
	
	...

	Attributes
	----------
	name : str
		the task unit name
	il : int
		instructions lines in the task * 10^factor
	factor : int
		factor of instruction lines

	Methods
	-------
	getil()
		returns number of instruction lines (*10^8 default) needed to complete the task
	"""

	def __init__(self, name="default_task", il=200, factor=8):
		"""
		Parameters
		----------
		name : str
			The name of the node core
		il : int
			Number of instruction lines per task * 10^factor
		factor : int
			factor for number of instruction lines
		"""

		# set up the attributes
		self.name = name
		self.il = il
		self.factor = factor

	def getil():
		""" Returns number of instruction lines (*10^self.factor) for this task
		"""
		return self.il

	def getfactor():
		""" Returns the factor (*10^factor), of the IL
		"""
		return self.factor