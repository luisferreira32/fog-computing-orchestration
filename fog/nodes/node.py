# external imports
from queue import Queue # O(1) FIFO for CPU queue
import math

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
	placement : int tuple
		the placement in space of the node
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

	def __init__(self, name="default_node", cpi=5, placement=(0,0)):
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
		self.placement = placement

		# and debug if set to do so
		if configs.FOG_DEBUG:
			print("[DEBUG] Node core "+self.name+" created: "+str(self.__dict__))


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
			task = self.cpuqueue.get(block=False)
		except Exception as EmptyCpuQueue:
			raise EmptyCpuQueue

		cycles = task.il*self.cpi

		if configs.FOG_DEBUG:
			print("[DEBUG] Finished processing task"+task.name+" with IL " + str(task.il)+"*10^"+str(task.factor) + " at node " + self.name)

		return cycles

	def queue(self, task = None):
		"""Add a task to the cpu queue

		Fails if no task is given

		Parameters
		------
		task=None
			is a task of type task.Unit to be processed

		Raises
		------
		EmptyTask
			if none was given to queue
		FullCpuQueue
			If there is no room in the cpu queue.
		"""
		if task is None:
			raise EmptyTask
		try:
			self.cpuqueue.put(task,block=False)
		except Exception as FullCpuQueue:
			raise FullCpuQueue

		if configs.FOG_DEBUG:
			print("[DEBUG] Added task"+task.name+" with IL " + str(task.il)+"*10^"+str(task.factor) + " to node " + self.name)


#------------------------------------------------------ Communication related ------------------------------------------------------

	def addcoms(self, coms=None):
		"""Add communication ability to the fog node

		Fails if no coms are given

		Raises
		------
		EmptyComs
			if none was given to queue
		"""
		if coms is None:
			raise EmptyComs
		self.coms = coms


	def getB(self):
		""" Getter for the communication bandwidth [MHz]
		"""
		if self.coms is None:
			return 0
		return self.coms.bandwidth


	def getP(self):
		""" Getter for the communication power [dBm]
		"""
		if self.coms is None:
			return 0
		return self.coms.power




#------------------------------------------------------ ------------ -----------------------------------------------------
#--------------------------------------------------- Functions on nodes --------------------------------------------------
#------------------------------------------------------ ------------ -----------------------------------------------------
