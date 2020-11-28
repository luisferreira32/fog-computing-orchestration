#!/usr/bin/env python
"""This file contains core classes for a discrete event simulation.

An Event_queue will accept Event objects to add it to a deque, and
sort it by their execution time. It has auxiliary functions that will
help the discrete event simulator run the events, with an option to
reset the queue.
The Event abstract class might be used for any inheritance to simulate
events within the aforementioned Event_queue.
"""

# >>>>> imports
# external imports
from collections import deque
from abc import ABC, abstractmethod

# our imports
from sim_env.configs import SIM_TIME

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> classes and functions
class Event(ABC):
	"""Event abstract class implementes the basic event form for a discrete event simulator.

	Necessary to include a time as parameter within the simulation time, defined in sim_env.configs,
	and when instanciating it with a subclass it is possible to give it a classtype string name.
	The execute method must be written in the subclass and may or may not add further events to the
	event queue that should be passed as an argument.
	"""

	def __init__(self, time, classtype=None):
		"""
		Parameters
		----------
		time: float
			time of the event execution
		classtype: str = None
			the class type of the subclass
		"""
		self.time = time
		self.classtype = classtype

	@abstractmethod
	def execute(self, evq = None):
		""" Should execute the event

		Parameters
		----------
		evq: Event_queue = None
			the event queue of the discrete event simulator
		"""
		# executes current event and adds more events to the Event_queue if necessary
		pass

class Event_queue(object):
	"""Event_queue class implements a sorted deque to run events in a discrete event simulator.

	Each event inserted must be a subclass of the abstract class Event, and have a time within
	the simulation time, defined in sim_env.configs. This class also has the methods to add,
	remove and do various observations to the event queue. Note that if an event is poped,
	the evq assumes the clock moved to that point, not accepting any previous new events.

	Attributes
	----------
	q: deque
		the event queue of an arbitrary lenght
	current_time: float
		a time keeper to avoid inserting event in the past
	"""

	def __init__(self):
		self.q = deque()
		self.current_time = 0

	def __str__(self):
		return "evq" + str(len(self.q))

	def add_event(self, e: Event):
		"""Adds an event to the queue sorted from left (larger time) to right (shorter time)

		Parameters
		----------
		e: Event
			an event, subclass of Event
		"""
		if not isinstance(e, Event):
			return
		# only process events within sim time
		if e.time > SIM_TIME or e.time < self.current_time:
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
		"""Returns the first event to be processed on the queue """
		if not self.has_events():
			return None
		ev = self.q.pop()
		self.current_time = ev.time
		return ev

	def queue_size(self):
		"""Returns the queue size """
		return len(self.q)

	def has_events(self):
		"""Returns a bool indicating wether there are or not events in the queue """
		return len(self.q) > 0

	def first_time(self):
		"""Returns the first event to be processed clock time """
		if not self.has_events():
			return -1
		return self.q[-1].time

	def queue(self):
		"""Returns the queue itself """
		return self.q

	def reset(self):
		"""Resets the current queue"""
		self.current_time = 0
		self.q.clear()

# <<<<<