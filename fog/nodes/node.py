# external imports
from queue import Queue # O(1) FIFO for CPU queue

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
	mips : int
		million of instructions per second the node CPU can process
	cpuqueue : Queue
		implements a tasks CPU queue
	Methods
	-------
	process(task=None)
		processes a task in the cpu queue, returning
	"""

	def __init__(self, name="default_node", mips=10):
		"""
		Parameters
		----------
		name : str
			The name of the node core
		"""

		# set up the attributes
		self.name = name
		self.mips = mips
		self.cpuqueue = Queue(configs.MAX_QUEUE)

		# and debug if set to do so
		if configs.FOG_DEBUG:
			print("[DEBUG] Node core "+self.name+" created.")

	def process(self, task=None):
		"""Process the first task in the CPU queue

		It's aim is to calculate the resources consumption, that will be returned

		Parameters
		----------
		task : task.Unit
			a task to be processed

		Raises
		------
		EmptyCpuQueue
			If there is no task in the cpu queue.
		"""
		if task is None:
			try:
				task = self.cpuqueue.get(block=False)
			except Exception as EmptyCpuQueue:
				raise EmptyCpuQueue

		if configs.FOG_DEBUG:
			print("[DEBUG] Finished processing task"+task.name+" with IL " + str(task.il)+"*10^"+str(task.factor) + " at node " + self.name)