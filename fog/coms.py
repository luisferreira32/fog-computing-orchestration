# external imports
import math

# import necessary fog environment configurations
from . import configs
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

	def __init__(self, bandwidth=configs.DEFAULT_BANDWIDTH, power=configs.DEFAULT_POWER):
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

def transmissionrate(n1=None, n2=None):	
	"""Calculate the fog transmission rate given two nodes from n1 - > n2

	Fails with lack of arguments or invalid values

	Parameters
	----------
	n1=None
		origin node n1, using its com device
	n2=None
		recieving node

	Return
	------
	transmission rate, or -1 if failed
	"""
	if n1 is None or n2 is None or n1.coms is None:
		if configs.FOG_DEBUG == 1: print("[DEBUG] None argument in transmissionrate()")
		return -1

	if node.distance(n1,n2) == 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] No distance between nodes, infinite transmissionrate")
		return -1

	BHz = n1.getB() * (10**6) # bandwidth in Hz and not MHz
	P = n1.getP() # power in dBm
	try:
		r = BHz*math.log(1+(configs.B1_PATHLOSS*((node.distance(n1,n2))**(-configs.B2_PATHLOSS))*P)/(BHz*configs.N0))
		if configs.FOG_DEBUG == 1: print("[DEBUG] transmission rate calculated is %.10f" % r)
		return r
	except Exception as InvalidParameters:
		raise InvalidParameters


def comtime(w0=0, rate12 = 0):
	"""Calculate the fog transmission time number of offloaded tasks and com rate between nodes

	Fails with lack of arguments or invalid values

	Parameters
	----------
	w0=0
		number of offloaded tasks
	t1=None
		task type being transmitted
	rate12=0
		transmission rate between n1 and n2

	Return
	------
	transmission time, or -1 if failed
	"""
	if w0 <= 0 or rate12 <=0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Argument error in comtime()")
		return -1

	# 2*Mbytes*w0/rate
	c = 2*configs.DEFAULT_DATA*(10**6)*w0/rate12
	if configs.FOG_DEBUG == 1: print("[DEBUG] Com time is ", c)
	return c