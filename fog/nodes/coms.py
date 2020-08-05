# import necessary fog environment configurations
from .. import configs
from . import node

#------------------------------------------------------ ------------- -----------------------------------------------------
#--------------------------------------------------- Fog Node Coms Class --------------------------------------------------
#------------------------------------------------------ ------------- -----------------------------------------------------

class Coms(object):
	"""
	The communication central of a fog computing node
	
	...

	Attributes
	----------
	bandwidth : int
		attributed MHz for the node
	power : int
		dBm of transmission power of the node

	Methods
	-------
	"""

	def __init__(self, bandwidth=2, power=20):
		"""
		Parameters
		----------
		name : str
			The name of the node core
		"""

		# set up the attributes
		self.bandwidth = bandwidth
		self.power = power

		# and debug if set to do so
		if configs.FOG_DEBUG:
			print("[DEBUG] Node coms created: "+str(self.__dict__))


#------------------------------------------------------ ------------- -----------------------------------------------------
#------------------------------------------------ Functions on the network ------------------------------------------------
#------------------------------------------------------ ------------- -----------------------------------------------------

