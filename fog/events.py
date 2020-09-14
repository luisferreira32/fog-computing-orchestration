# external imports
import collections

# our configs
from . import configs

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
	def __init__(self, time):
		self.time = time


class Recieving(Event):
	"""docstring for Recieving"""
	def __init__(self, time):
		super(Recieving, self).__init__(time)


class Sending(Event):
	"""docstring for Sending"""
	def __init__(self, time):
		super(Sending, self).__init__(time)


class Processing(Event):
	"""docstring for Processing"""
	def __init__(self, time):
		super(Processing, self).__init__(time)

		
		
		