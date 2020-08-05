# external imports
from queue import Queue # O(1) FIFO for CPU queue
import math # for distance calculation

# import necessary fog environment configurations
from .. import configs
from .. import task

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
	cpuqueue : Queue
		implements a tasks CPU queue
	influx : int
		number of tasks allocated per second to this node
	wL : int
		number of tasks to be processed in this node
	placement : int tuple
		the placement in space of the node (x,y) [meter, meter]
	coms : Coms
		the communication center of the node

	Methods
	-------
	process()
		processes a task in the cpu queue, returning number of cycles needed
	queue(task=None)
		adds a task to the CPU queue
	addcoms(coms=None)
		adds communication ability to the node
	"""

	def __init__(self, name="default_node", cpi=5, placement=(0,0), coms=None):
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
		self.name = name
		self.cpi = cpi
		self.cpuqueue = Queue(configs.MAX_QUEUE)
		self.influx = 0
		self.wL = 0
		self.placement = placement
		self.coms = coms

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
		return 0

	def timestep(self, t1=None, w0=0): # TODO@luis: use a factor for timesteps to allow finer tunning
		"""Each timestep is a second, this function runs the second on this world
		
		Parameters
		----------
		t1=None
			configure the task type that is coming in, if not, the default is used
		w0 = 0
			number of tasks offloaded to another node

		Return
		------
		pending
			number of pending tasks, either to be offloaded or discarded
		"""

		# default task if none is given
		if t1 is None:
			t1 = task.Unit()

		# The number of tasks to be processed in this time step = previous pending + INFLUX - SERVICE RATE - w0
		self.wL += self.influx - configs.SERVICE_RATE - w0
		# more tasks solved than queued
		if self.wL < 0:
			while self.wL < 0:
				if self.cpuqueue.empty():
					self.wL = 0
					break
				self.process()
				self.wL+=1
		# perfect balance means nothing will be done, i.e. new allocated tasks, will be consumed at the service rate / offload
		elif self.wL == 0:
			pass
		# and if more influx than service rate, means we'll add to the queue
		else:
			while self.wL > 0:
				if self.cpuqueue.full():
					if configs.FOG_DEBUG: print("[DEBUG] overload on node "+self.name)
					return self.wL
				self.queue(t1)
				self.wL-=1
						

		# debug print
		if configs.FOG_DEBUG:
			print("[DEBUG] time step completed, queue size at "+str(self.cpuqueue.qsize())+" and influx at "+str(self.influx))

		return self.wL


#------------------------------------------------------ CPU related ------------------------------------------------------

	def process(self):
		"""Process the first task in the CPU queue

		Calculates the number of cycles elapsed on the CPU to solve the task

		Raises
		------
		EmptyCpuQueue
			If there is no task in the cpu queue.
		"""
		try:
			t1 = self.cpuqueue.get(block=False)
		except Exception as EmptyCpuQueue:
			raise EmptyCpuQueue

		cycles = t1.il*self.cpi

		if configs.FOG_DEBUG:
			print("[DEBUG] Finished processing task"+t1.name+" with IL " + str(t1.il)+"*10^"+str(t1.factor) + " at node " + self.name)

		return cycles

	def queue(self, t1 = None):
		"""Add a task to the cpu queue

		Parameters
		------
		t1=None
			is a task of type task.Unit to be processed, default_task if none given

		Raises
		------
		FullCpuQueue
			If there is no room in the cpu queue.
		"""
		if t1 is None:
			t1 = task.Unit()

		try:
			self.cpuqueue.put(t1,block=False)
		except Exception as FullCpuQueue:
			print("[DEBUG] Queue overloaded one node "+self.name)
			raise FullCpuQueue

		if configs.FOG_DEBUG:
			print("[DEBUG] Added task ("+t1.name+") with IL " + str(t1.il)+"*10^"+str(t1.factor) + " to node " + self.name)
		return 0


#------------------------------------------------------ Communication related ------------------------------------------------------


	def addcoms(self, coms=None):
		"""Add communication ability to the fog node

		Fails if no coms are given

		"""
		if coms is None:
			return -1
		self.coms = coms
		return 0


	def getB(self):
		""" Getter for the communication bandwidth [MHz]
		"""
		if self.coms is None:
			return None
		return self.coms.bandwidth


	def getP(self):
		""" Getter for the communication power [dBm]
		"""
		if self.coms is None:
			return None
		return self.coms.power




#------------------------------------------------------ ------------ -----------------------------------------------------
#--------------------------------------------------- Functions on nodes --------------------------------------------------
#------------------------------------------------------ ------------ -----------------------------------------------------

# - frequency calculator based on service rate

def getfreq(cycles=None):
	"""Calculate clock frequency  based on the number of cycles a task takes and the configured service rate

	Fails if no cycles are given, or if the service rate is miss configured

	Return
	------
	clock frequency [Hz] * 10 ^ task.factor
		frequency of the clock if successful
		-1 if unsuccessful
	"""
	if cycles is None or configs.SERVICE_RATE <= 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Empty cycles or bad service rate in getfreq()")
		return -1

	return configs.SERVICE_RATE*cycles



# - task execution time per task based on service rate [IRRELEVANT FUNCTION, it's the inverse of SERVICE RATE]

def extime(sr=configs.SERVICE_RATE):
	"""Calculate the task execution time, based on the SERVICE RATE

	Fails if the service rate is miss configured

	Return
	------
	execution time [s] 
		or -1 if failed
	"""
	if sr <= 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Bad service rate in extime()")
		return -1

	return 1/sr

# - distance calculator based on nodes placement

def distance(n1=None, n2=None):
	"""Calculate the distance between two nodes on a plane

	Fails if either node is none returning -1

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