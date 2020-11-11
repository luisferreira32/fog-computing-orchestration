#!/usr/bin/env python

# external imports
import collections
import sys
from abc import ABC, abstractmethod

# our imports
from sim_env.configs import SIM_TIME
from sim_env.calculators import bernoulli_arrival
from sim_env.core_classes import Task, task_processing_time, task_communication_time

# -------------------------------------------- Event Queue --------------------------------------------

class Event_queue(object):
	# handles queueing and sorting of events
	def __init__(self):
		# arbitrary lenghted queue
		self.q = collections.deque()

	def __str__(self):
		return "evq" + str(len(self.q))

	def queueSize(self):
		return len(self.q)

	def hasEvents(self):
		return len(self.q) > 0

	def addEvent(self, e):
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
		return

	def popEvent(self):
		return self.q.pop()

	def first_time(self):
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

class Set_arrivals(Event):
	""" Set_arrivals calculates which nodes and slices are recieving a task this timestep
	"""
	def __init__(self, time, timestep, nodes):
		super(Set_arrivals, self).__init__(time, "Set_arrivals")
		self.timestep = timestep
		self.nodes = nodes

	def execute(self, evq):
		# for each node slice, check wether the task arrived or not, and place an event for the next timestep
		for n in self.nodes:
			for i in range(n.max_k):
				if bernoulli_arrival(n._arrivals_on_slices[i]):
					t = Task(self.time+self.timestep, task_type=n._task_type_on_slices[i])
					ev = Task_arrival(self.time+self.timestep, n, i, t)
					evq.addEvent(ev)
		# then recursevly ask for another set of arrivals
		evq.addEvent(Set_arrivals(self.time+self.timestep, self.timestep, self.nodes))
		return None

class Task_arrival(Event):
	""" Task_arrival inserts a task to a node slice, if it overflows returns the task
	"""
	def __init__(self, time, node, k, task):
		super(Task_arrival, self).__init__(time, "Task_arrival")
		self.node = node
		self.k = k
		self.task = task
		assert time >= task.task_time()

	def execute(self, evq):
		# if the task just arrived, schedule a discard for it's deadline (in milliseconds) # NOTE: only when deciding for now
		# evq.addEvent(Discard_task(self.task.task_time()+(0.001*self.task.delay_constraint), self.node, self.k, self.task))
		return self.node.add_task_on_slice(self.k, self.task)

	def task_time(self):
		return self.task.task_time()
		
class Task_finished(Event):
	""" Task_finished specifies a task finished processing and returns it
	"""
	def __init__(self, time, node, k, task):
		super(Task_finished, self).__init__(time, "Task_finished")
		self.node = node
		self.k = k
		self.task = task

	def execute(self, evq):
		self.task.finish_processing(self.time)
		self.node._dealt_tasks += 1
		return self.node.remove_task_of_slice(self.k, self.task)

class Start_processing(Event):
	""" Start_processing executes the task of starting to process w tasks
	"""
	def __init__(self, time, node, k, w):
		super(Start_processing, self).__init__(time, "Start_processing")
		self.node = node
		self.k = k
		self.w = w
		
	def execute(self, evq):
		tasks_under_processing = self.node.start_processing_in_slice(self.k, self.w)
		# discard and set finish processing when decisions are made
		for task in tasks_under_processing:
			finish = self.time+task_processing_time(task)
			if task.exceeded_contraint(finish): # TODO@luis: redo this on another place
				evq.addEvent(Discard_task(task.constraint_time(), self.node, self.k, task))
			else:
				evq.addEvent(Task_finished(finish, self.node, self.k, task))
		return None

class Offload(Event):
	""" Offloads the task that just arrived to a destination node
	"""
	def __init__(self, time, node, k, destination, con=1):
		super(Offload, self).__init__(time, "Offload")
		self.node = node
		self.k = k
		self.destination = destination
		self.concurrent_offloads = con

	def execute(self, evq):
		# can't send if there is no way to send or it's busy sending
		if self.node._communication_rates[self.destination.index-1] == 0: return None
		if self.node.transmitting: return None
		# then pop the last task we got
		t = self.node.pop_last_task(self.k, self.time)
		# if it's an invalid choice return without sending out the task
		if t == None: return None
		# else plan the landing
		self.node._dealt_tasks += 1
		self.node.transmitting = True
		arrive_time = self.time + task_communication_time(t, self.node._communication_rates[self.destination.index-1]/self.concurrent_offloads)
		evq.addEvent(Task_arrival(arrive_time, self.destination, self.k, t))
		evq.addEvent(Finished_transmitting(arrive_time, self.node))
		return None

class Finished_transmitting(Event):
	""" Finished_transmitting is an event that sets the transmission flag down
	"""
	def __init__(self, time, node):
		super(Finished_transmitting, self).__init__(time, "Finished_transmitting")
		self.node = node

	def execute(self,evq):
		self.node.transmitting = False
		return None

class Discard_task(Event):
	"""Discard_task that has its' delay constraint unmet
	"""
	def __init__(self, time, node, k, task):
		super(Discard_task, self).__init__(time, "Discard_task")
		self.node = node
		self.k = k
		self.task = task

	def execute(self, evq):
		return self.node.remove_task_of_slice(self.k, self.task)
		

