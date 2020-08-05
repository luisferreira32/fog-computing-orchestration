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
	cps : int
		cpu cycles per second (cpu frequency) * 10 ^8
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

	def __init__(self, name="default_node", cpi=configs.DEFAULT_CPI, cps=1800, placement=(0,0), influx=configs.TASK_ARRIVAL_RATE,coms=None):
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
		self.cps = cps
		self.cpuqueue = Queue(configs.MAX_QUEUE)
		self.influx = influx
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

	def recieveoffload(self, offloaded=0):
		"""Adds the offloaded tasks to the tasks to be processed

		Parameters
		----------
		offloaded=0
			number of offloaded tasks to this node
		"""
		self.wL += offloaded

	def timestep(self, t1=task.Unit(), w0=0):
		"""Each timestep is a second, this function runs the second on this world
		
		Parameters
		----------
		t1=None
			configure the task type that is coming in, if not, the default is used
		w0 = 0
			number of tasks offloaded to another node

		Return
		------
		remaining time of the timestep < 1s
		"""

		# each second we have N cpu cycles to use on tasks
		cycles_remaining = self.cps
		# The number of tasks to be processed in this node = previous pending + INFLUX - w0 (out flux)
		self.wL += self.influx - w0

		if not self.cpuqueue.empty():
			task_cycles = self.process()
			if task_cycles > cycles_remaining:
				return cycles_remaining/self.cps # task delayed for next timestep, and this one is cut shorter
		
		# debug print
		if configs.FOG_DEBUG:
			print("[DEBUG] time step completed, queue size at "+str(self.cpuqueue.qsize())+" and influx at "+str(self.influx))

		return cycles_remaining/self.cps


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
			print("[DEBUG] Finished processing task"+t1.name+" with IL " + str(t1.il)+"*10^8 at node " + self.name)

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
			print("[DEBUG] Added task ("+t1.name+") with IL " + str(t1.il)+"*10^8 to node " + self.name)
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

def avgcps(n=Core(), t=task.Unit(), sr=configs.SERVICE_RATE):
	"""Calculate cycles per second (*10^8) considering a model node, model task and a model service rate
	"""
	return sr*n.cpi*t.il



# - task execution time on node local and offloaded

def extime(n1=None, n2=None, w0=0, t1= task.Unit()):
	"""Calculate task execution time on node local and offloaded

	Return
	------
	execution time [s] 
		or -1 if failed
	"""
	if n1 is None or n2 is None:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Invalid parameters in extime()")
		return -1

	return 1/sr


# - task waiting time on node

def wtime(sr=configs.SERVICE_RATE):
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