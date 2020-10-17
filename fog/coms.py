# external imports
import math

# import necessary fog environment configurations
from . import configs
from . import node

# ------------------------------------------------------ Comunication related classes ------------------------------------------------------

class Task(object):
	def __init__(self, timestamp, size=configs.DEFAULT_DATA, instruction_lines=configs.DEFAULT_IL):
		self.timestamp = timestamp
		self.size = size
		self.il = instruction_lines
		self.completed = False
		self.delay = -1

	def process(self, finishing_time):
		self.completed = True
		self.delay = finishing_time - self.timestamp
		return self.delay

		
#------------------------------------------------------ ------------- -----------------------------------------------------
#------------------------------------------------ Functions on the network ------------------------------------------------
#------------------------------------------------------ ------------- -----------------------------------------------------

def transmissionrate(n1=None, n2=None, bw=None, pw=None):	
	# Calculate the fog transmission rate given two nodes from n1 - > n2

	if n1 is None or n2 is None:
		if configs.FOG_DEBUG == 1: print("[DEBUG] None argument in transmissionrate()")
		return -1

	if node.distance(n1,n2) == 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] No distance between nodes",n1.name, n2.name)
		return -1
 	
 	# MHz to Hz
	BHz = bw * (1000000)
	# convertion from dBm to mW in power and noise
	Power = math.pow( 10 , 0.1*pw)*(configs.B1_PATHLOSS*((node.distance(n1,n2))**(-configs.B2_PATHLOSS))) 
	Noise = math.pow( 10 , 0.1*configs.N0)*BHz
	try:
		# Shannon-hartley theorem ~: r = B log2 ( 1 + SNR )
		r = BHz*math.log2(1 + (Power/ Noise))
		if configs.FOG_DEBUG == 1: print("[DEBUG] transmission rate calculated Mb/s",r/1000000)
		return r
	except Exception as InvalidParameters:
		raise InvalidParameters


def comtime(w0=0, rate12 = 0):
	# Calculate the fog transmission time number of offloaded tasks and com rate between nodes
	if w0 <= 0 or rate12 <=0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Argument error in comtime()")
		return -1

	# 2*Mbytes*w0/rate
	c = 2*(configs.DEFAULT_DATA*(1000000))*w0/rate12
	if configs.FOG_DEBUG == 1: print("[DEBUG] Com time is ", c)
	return c