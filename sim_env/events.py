# external imports
import collections
import sys
from abc import ABC, abstractmethod

# our imports
from sim_env.configs import SIM_TIME_STEPS
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
		if e.time > SIM_TIME_STEPS or e.time < 0:
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

class Task_arrival(Event):
	""" Task_arrival inserts a task to a node slice, if it overflows returns the task
	"""
	def __init__(self, time, node, k, task):
		super(Task_arrival, self).__init__(time, "Task_arrival")
		self.node = node
		self.k = k
		self.task = task

	def execute(self, evq):
		# should schedule the deadline event if deadline is implemented
		return self.node.add_task_on_slice(self.k, self.task)
		
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
		return self.node.remove_task_of_slice(k,self.node, self.task)

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
		for task in tasks_under_processing:
			finish = self.time+task_processing_time(task)
			evq.addEvent(Task_finished(finish, self.node, self.k, task))

class Offload(Event):
	""" Offloads the task that just arrived to a destination node
	"""
	def __init__(self, time, node, k, destination):
		super(Offload, self).__init__(time)
		self.node = node
		self.k = k
		self.destination = destination

	def execute(self, evq):
		# can't send if there is no way to send
		if self.node._communication_rates[self.destination.index] == 0: return
		# then pop the last task we got
		t = self.node.pop_last_task(self.k, self.time)
		# if it's an invalid choice return empty handed
		if t == None: return
		# else plan the landing
		self.node._dealt_tasks += 1
		arrive = self.time + task_communication_time(t, self.node._communication_rates[self.destination.index])
		evq.addEvent(Task_arrival(arrive, self.destination, self.k, t))


		

