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
	clock : float
		internal cpu time [s]

	Methods
	-------
	process()
		processes a task in the cpu queue, returning number of cycles needed
	queue(task=None)
		adds a task to the CPU queue
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


	def setinflux(self, newinflux=0):
		"""Sets the influx of a node, i.e. tasks allocated per second
		"""
		if newinflux > configs.MAX_INFLUX:
			return -1
		self.influx = newinflux
		return self.influx

	def excessinflux(self, recieved=None):
		"""Calculates excess influx in this timestep, only redirect if it's our own tasks
		"""
		if self.influx > configs.MAX_QUEUE - self.cpuqueue.qsize() - recieved.qsize():
			return self.influx - max(configs.MAX_QUEUE - self.cpuqueue.qsize() - recieved.qsize(), 0)
		return 0


	#------------------------------------------------------ CPU related ------------------------------------------------------

	def emptyqueue(self):
		"""Checks if CPU queue is empty
		"""
		return self.cpuqueue.empty()

	def fullqueue(self):
		"""Checks if CPU queue is full
		"""
		return self.cpuqueue.full()

	def qs(self):
		"""Checks CPU queue size
		"""
		return self.cpuqueue.qsize()

	def process(self, time=0):
		"""Process the first task in the CPU queue: fractions of a second for processing a task

		Parameters
		----------
		given time for processing

		Returns
		-------
		list of processed tasks delays
		"""

		# we can't accumulate time we're not technically "using" before the next time step
		if self.cpuqueue.empty():
			self.clock += time
			time = 0
		if time >= 2:
			self.clock += 1
			time -= 1

		solved = []
		while not self.cpuqueue.empty() and time - configs.DEFAULT_IL*self.cpi/self.cps >= 0:
			time -= configs.DEFAULT_IL*self.cpi/self.cps
			self.clock += configs.DEFAULT_IL*self.cpi/self.cps
			solved.append(self.clock - self.cpuqueue.get(False))

		if configs.FOG_DEBUG:
			print("[DEBUG] Node "+ self.name +" cpu timer excess is %.2f and queue size %d" % (float(time), self.cpuqueue.qsize()))
			if self.cpuqueue.empty(): print("[DEBUG] No task to process at node " + self.name)

		return solved

	def queue(self, recieved=None,offloaded=0):
		"""Add a task to the cpu queue and set the locally processed tasks number

		Parameters
		----------
		recieved=None
			queue with incoming tasks
		offloaded=0
			number of offloaded tasks

		Return
		------
		number of discarded tasks
		"""

		# only count the recieved if there's actually recieved tasks
		if recieved is None:
			recieved = queue.Queue() # empty queue

		# figure out how many we're working locally
		self.wL = self.influx + recieved.qsize() - offloaded
		mytasks = self.influx - offloaded
		
		while not self.cpuqueue.full() and mytasks >= 1:
			self.cpuqueue.put(self.clock, False)
			mytasks -= 1

		while not self.cpuqueue.full() and not recieved.empty():
			# place a task with a com delay already
			self.cpuqueue.put(recieved.get(False), False)

		if configs.FOG_DEBUG and self.fullqueue():
			print("[DEBUG] Full queue at node " + self.name)

		discarded = mytasks+recieved.qsize()
		self.wL -= discarded
		if configs.FOG_DEBUG and discarded > 0: print("[DEBUG] Node",self.name,"discarded",discarded,"tasks with",recieved.qsize(),"from other nodes")

		return discarded


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