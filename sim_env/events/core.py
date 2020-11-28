#!/usr/bin/env python

# external imports
import collections
import sys
from abc import ABC, abstractmethod

# our imports
from sim_env.configs import SIM_TIME

# -------------------------------------------- Event Queue --------------------------------------------

class Event_queue(object):
	# handles queueing and sorting of events
	def __init__(self):
		# arbitrary lenghted queue
		self.q = collections.deque()

	def __str__(self):
		return "evq" + str(len(self.q))

	def queue_size(self):
		return len(self.q)

	def has_events(self):
		return len(self.q) > 0

	def add_event(self, e):
		# only process events within sim time
		if e.time > SIM_TIME or e.time < 0:
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

	def pop_event(self):
		return self.q.pop()

	def first_time(self):
		if not self.has_events():
			return -1
		return self.q[-1].time

	def queue(self):
		return self.q

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


# -- aux functions --

def is_arrival_on_slice(ev, node, k):
	if ev.classtype == "Task_arrival" and ev.node == node and ev.k == k:
		return True
	return False

def is_offload_arrival_event(ev, time):
	if ev.classtype == "Task_arrival" and ev.task_time() < clock:
		return True
	return False