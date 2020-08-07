# external imports
import math # for distance calculation
import queue

# import necessary fog environment configurations
from . import configs

#------------------------------------------------------ ------------- -----------------------------------------------------
#--------------------------------------------------- Fog Node Core Class --------------------------------------------------
#------------------------------------------------------ ------------- -----------------------------------------------------

class Core(object):
	"""
	The core of a fog computing node has all its attributes.
	
	...

	Attributes
	----------
	name : str
		the fog node name
	cpi : int
		cpu cycles per instruction
	cps : int
		cpu cycles per second (cpu frequency) * 10 ^8
	cpuqueue : int
		implements a tasks CPU queue
	influx : int
		number of tasks allocated per second to this node
	wL : int
		number of tasks to be processed in this node
	placement : int tuple
		the placement in space of the node (x,y) [meter, meter]
	coms : dictionary
		the communication parameters of the node
	clock : int
		internal cpu time [s]

	Methods
	-------
	process()
		processes a task in the cpu queue, returning number of cycles needed
	queue(task=None)
		adds a task to the CPU queue
	addcoms(coms=None)
		adds communication ability to the node
	"""

	def __init__(self, name="default_node", cpu=(configs.DEFAULT_CPI, configs.DEFAULT_CPS), placement=(0,0), influx=configs.TASK_ARRIVAL_RATE, 
		coms=(configs.DEFAULT_POWER, configs.DEFAULT_BANDWIDTH)):
		"""
		Parameters
		----------
		name : str
			The name of the node core
		cpi : int
			cycles per instruction
		placement : (int, int)
			placement in space
		"""

		# set up the attributes
		# generic
		self.name = name
		self.influx = influx
		self.wL = 0
		self.placement = placement
		# cpu related
		self.cpi = cpu[0]
		self.cps = cpu[1]
		self.cpuqueue = queue.Queue(maxsize=configs.MAX_QUEUE)
		self.clock = 0
		# comunication related
		self.coms = {
			"power": coms[0],
			"bandwidth" : coms[1]
		}

		# and debug if set to do so
		if configs.FOG_DEBUG:
			print("[DEBUG] Node core "+self.name+" created: "+str(self.__dict__))


	def addinflux(self, newinflux=0):
		"""Adds tasks influx to the node

		If maximum influx is exceeded, return -1 and don't add it
		"""
		if self.influx+newinflux > configs.MAX_INFLUX:
			return -1
		self.influx = self.influx+newinflux
		return self.influx

	def excessinflux(self, recieved=0, offloaded=0):
		"""Calculates excess influx in this timestep
		"""
		if self.influx > configs.MAX_QUEUE - self.cpuqueue.qsize() - recieved + offloaded:
			return self.influx - (configs.MAX_QUEUE - self.cpuqueue.qsize() - recieved + offloaded)
		return 0

	def setwL(self, recieved=0,offloaded=0):
		"""Sets the number of tasks to be locally processed in this time step

		Parameters
		----------
		recieved=0
			number of tasks recieved by other nodes offloading
		offloaded=0
			number of offloaded tasks to offload of this node

		Returns
		-------
		number of discarded tasks
		"""
		self.wL = self.influx + recieved - offloaded
		discarded = 0
		if self.wL > configs.MAX_QUEUE - self.cpuqueue.qsize():
			discarded = self.wL - (configs.MAX_QUEUE - self.cpuqueue.qsize())
			self.wL = self.wL - discarded
		if configs.FOG_DEBUG:
			if discarded > 0: print("[DEBUG] wL Discarded",discarded," tasks at node " + self.name)
			if discarded == 0 and self.wL != 0: print("[DEBUG] wL No task discarded at node " + self.name)
		return discarded


	#------------------------------------------------------ CPU related ------------------------------------------------------

	def emptyqueue(self):
		"""Checks if CPU queue is empty
		"""
		return self.cpuqueue.empty()

	def fullqueue(self):
		"""Checks if CPU queue is full
		"""
		return self.cpuqueue.full()

	def process(self, time=0):
		"""Process the first task in the CPU queue: fractions of a second for processing a task

		Parameters
		----------
		given time for processing

		Returns
		-------
		list of processed tasks delays
		"""
		solved = []
		while not self.cpuqueue.empty() and time - configs.DEFAULT_IL*self.cpi/self.cps > 0:
			time -= configs.DEFAULT_IL*self.cpi/self.cps
			self.clock += configs.DEFAULT_IL*self.cpi/self.cps
			solved.append(self.cpuqueue.get(False))

		# we can't accumulate time we're not technically "using" before the next time step
		if self.cpuqueue.empty():
			self.clock += time

		if configs.FOG_DEBUG:
			print("[DEBUG] Node "+ self.name +" cpu timer excess is %.2f and queue size %d" % (float(time), self.cpuqueue.qsize()))
			if self.cpuqueue.empty(): print("[DEBUG] No task to process at node " + self.name)

		return time

	def queue(self):
		"""Add a task to the cpu queue according to tasks to be processed locally wL

		Return
		------
		queue size
		"""
		while not self.cpuqueue.full() and self.wL >= 1:
			self.cpuqueue.put(self.clock, False)
			self.wL -= 1
			if configs.FOG_DEBUG:
				print("[DEBUG] Queued task to node " + self.name+" to Q size",self.cpuqueue.qsize())
		if configs.FOG_DEBUG and self.fullqueue():
			print("[DEBUG] Full queue at node " + self.name)

		return self.cpuqueue.qsize()


	#------------------------------------------------------ Communication related ------------------------------------------------------


	def getB(self):
		""" Getter for the communication bandwidth [MHz]
		"""
		return self.coms.get("bandwidth")


	def getP(self):
		""" Getter for the communication power [dBm]
		"""
		return self.coms.get("power")




#------------------------------------------------------ ------------ -----------------------------------------------------
#--------------------------------------------------- Functions on nodes --------------------------------------------------
#------------------------------------------------------ ------------ -----------------------------------------------------

# - frequency calculator based on service rate

def avgcps(n=Core(), sr=configs.SERVICE_RATE):
	"""Calculate cycles per second (*10^8) considering a model node, model task and a model service rate
	"""
	return sr*n.cpi*configs.DEFAULT_IL



# - task execution time on node local and offloaded

def extime(n1=None, n2=None, w0=0):
	"""Calculate task execution time on node local and offloaded
	
	Parameters
	----------
	n1=None
		current node
	n2=None
		dest node
	w0=0
		number of offloaded tasks

	Return
	------
	execution time [s] 
		or -1 if failed
	"""
	if n1 is None or n2 is None or w0 < 0 or n1.wL < 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Invalid parameters in extime()")
		return -1

	return ((configs.DEFAULT_IL*n1.cpi*n1.wL)/n1.cps + (configs.DEFAULT_IL*n2.cpi*w0)/n2.cps)


# - task waiting time on node

def wtime(n1=None, n2=None, w0=0):
	"""Calculate the average waiting time
	
	Parameters
	----------
	n1=None
		current node
	n2=None
		dest node
	w0=0
		number of offloaded tasks

	Return
	------
	avg waiting time time [s] 
		or -1 if failed
	"""
	if n1 is None or n2 is None or w0 < 0 or n1.wL < 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Invalid parameters in wtime()")
		return -1

	wt = 0
	if n1.wL > 0: wt += n1.cpuqueue.qsize()/configs.SERVICE_RATE
	if w0 > 0: wt += n1.cpuqueue.qsize()/configs.SERVICE_RATE + n2.cpuqueue.qsize()/configs.SERVICE_RATE
	# wt = (QL/srL)[if wL != 0] + (QL/srL + Q0/sr0)[if w0 != 0]
	return wt

# - distance calculator based on nodes placement

def distance(n1=None, n2=None):
	"""Calculate the distance between two nodes on a plane

	Fails if either node is none returning -1

	Parameters
	----------
	n1=None
		current node
	n2=None
		dest node

	Return
	------
	distance between two nodes in meters
	"""
	if n1 is None or n2 is None:
		if configs.FOG_DEBUG == 1: print("[DEBUG] None argument in distance()")
		return -1
	try:
		return math.sqrt((n1.placement[0] - n2.placement[0])**2 + (n1.placement[1] - n2.placement[1])**2)
	except Exception as InvalidParameters:
		raise InvalidParameters