# external imports
import collections

# our configs
from . import configs
from . import coms


class EventQueue(object):
	# handles queueing and sorting of events
	def __init__(self):
		# arbitrary lenghted queue
		self.q = collections.deque()
		self.recieving_client = 0

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
	def __init__(self, time, classtype=None):
		self.time = time
		self.classtype = classtype


class Recieving(Event):
	def __init__(self, time, recieving_node, incoming_task, decision=None, sending_node=None):
		super(Recieving, self).__init__(time, "Recieving")
		self.rn = recieving_node
		self.it = incoming_task

		# a set for the decision for the next "w" tasks: [n0, w0] with w0 < w
		self.decision = decision
		self.sn = sending_node

	# allocs a task to node queue, offloads to another or discards.
	def execute(self, eq):
		# if it comes from another offloading
		if self.decision is None and self.sn is not None:
			self.sn.sending = False
			return self.rn.queue(self.it)

		# if we're meant to offload and we can... do it
		if self.decision["w0"] > 0 and not self.rn.sending():
			self.decision["w0"] = self.decision["w0"] - 1
			ev = Sending(self.time, self.rn, self.decision["n0"], self.it)
			eq.addEvent(ev)
			t = None
		else:
			# queue the task if we didn't offload it, returns task if queue is full
			t = self.rn.queue(self.it)

		# and start processing if it hasn't started already
		if not self.rn.processing:
			ev = Processing(self.time, self.rn)
			eq.addEvent(ev)

		return t


class Sending(Event):
	def __init__(self, time, sending_node, recieving_node, outbound_task):
		super(Sending, self).__init__(time, "Sending")
		self.sn = sending_node
		self.rn = recieving_node
		self.ot = outbound_task

	# sends the task to another node, blocking coms in the meantime
	def execute(self, eq):
		self.sn.sending = True
		# recieves after comm time finished
		ev = Recieving(self.time+self.sn.comtime[self.rn.name], self.rn,self.ot)
		eq.addEvent(ev)



class Processing(Event):
	def __init__(self, time, processing_node):
		super(Processing, self).__init__(time, "Processing")
		self.pn = processing_node

	# processes first task in node queue and sets other processing events if there's still space
	def execute(self, eq):
		self.pn.processing = True
		t = self.pn.process(self.time)
		# if there was no task to process, we finished processing, so go false
		if t is None:
			self.pn.processing = False
		else:
			self.pn.processing = True
			# p_time is when the task t processing has finished, so starts next processing event
			p_time = t.delay + t.timestamp
			ev = Processing(p_time, self.pn)
			eq.addEvent(ev)
		return t		
		
		