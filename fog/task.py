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
		instructions lines in the task * 10^8
	data : int
		the data size of a task [Mbytes]

	"""

	def __init__(self, name="default_task", il=configs.DEFAULT_IL, data=configs.DEFAULT_DATA):
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
		self.data = data
	