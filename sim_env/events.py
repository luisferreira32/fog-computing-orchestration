# external imports
import collections
import sys
from abc import ABC, abstractmethod

# our imports

# -------------------------------------------- Event Queue --------------------------------------------

class EventQueue(object):
	# handles queueing and sorting of events
	def __init__(self):
		# arbitrary lenghted queue
		self.q = collections.deque()

	def __str__(self):
		return "Event Queue of len" + str(len(self.q))

	def queueSize(self):
		return len(self.q)

	def hasEvents(self):
		return len(self.q) > 0

	def addEvent(self, e):
		# only process events within sim time
		if e.time > configs.SIM_TIME or e.time < 0:
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

	def first_time(self):
		return self.q[-1].time

	def reset(self):
		self.q.clear()


# -------------------------------------------- Events --------------------------------------------

class Event(ABC):
	def __init__(self, time, classtype=None):
		self.time = time
		self.classtype = classtype

	@abstractmethod
	def execute(self, evq):
		# executes current event and adds more events to the EVentQueue
		pass

