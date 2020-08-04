# external imports
from queue import Queue # O(1) FIFO for CPU queue
import math

# import necessary fog environment configurations
from .. import configs
from .. import task

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

	Methods
	-------
	process(task=None)
		processes a task in the cpu queue, returning
	"""

	def __init__(self, name="default_node", cpi=5, placement=(0,0)):
		"""
		Parameters
		----------
		name : str
			The name of the node core
		"""

		# set up the attributes
		self.name = name
		self.cpi = cpi
		self.cpuqueue = Queue(configs.MAX_QUEUE)
		self.placement = placement

		# and debug if set to do so
		if configs.FOG_DEBUG:
			print("[DEBUG] Node core "+self.name+" created: "+str(self.__dict__))

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
