# external imports
import collections

# our configs
from . import configs

# !!!! TODO@LUIS: commet all this classes properly !!!!


class EventQueue(object):
	"""docstring for EventQueue"""
	def __init__(self):
		# arbitrary lenghted queue
		self.q = collections.deque()

	def __str__(self):
		times = []
		for e in self.q:
			times.append(e.time)
		return str(times)

	def queueSize(self):
		return len(self.q)

	def addEvent(self, e):
		# only process events within sim time
		if e.time > configs.SIM_TIME:
			return
		# if there's no events just add it
		if len(self.q) == 0:
			self.q.append(e)
			return

		# insert in an orderly fashion time decreases from left to right, for faster insert
		for ev in self.q:
			if ev.time > e.time:
				continue
			ind = self.q.index(ev)
			self.q.insert(ind, e)
			return
		self.q.append(e)
		return

	def popEvent(self):
		return self.q.pop()
		
class Event(object):
	"""docstring for Event"""
	def __init__(self, time, classtype=None):
		self.time = time
		self.classtype = classtype


class Recieving(Event):
	"""docstring for Recieving"""
	def __init__(self, time, recieving_node, incoming_task, algorithm, next_interval=None):
		super(Recieving, self).__init__(time, "Recieving")
		self.rn = recieving_node
		self.it = incoming_task
		self.al = algorithm
		# the interval til next task arrives
		self.ni = next_interval

	# allocs a task to node queue, offloads to another or discards.
	def execute(self):
		pass


class Sending(Event):
	"""docstring for Sending"""
	def __init__(self, time, sending_node, outbound_task):
		super(Sending, self).__init__(time, "Sending")
		self.sn = sending_node
		self.ot = outbound_task

	# sends the task to another node, blocking coms in the meantime
	def execute(self):
		pass


class Processing(Event):
	"""docstring for Processing"""
	def __init__(self, time, processing_node):
		super(Processing, self).__init__(time, "Processing")
		self.pn = processing_node

	# processes first task in node queue and sets other processing events if there's still space
	def execute(self):
		pass

		
		
		