# external imports
import collections
import sys

# our imports
from . import configs
from . import coms
from tools import utils, graphs
from algorithms import basic

# -------------------------------------------- Event Queue --------------------------------------------

class EventQueue(object):
	# handles queueing and sorting of events
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


# -------------------------------------------- Events --------------------------------------------

class Event(object):
	def __init__(self, time, classtype=None):
		self.time = time
		self.classtype = classtype


# -------------------------------------------- Decision --------------------------------------------

class Decision(Event):
	def __init__(self, time, nL, nodes, algorithm="rd", time_interval = configs.TIME_INTERVAL, 
		ar=configs.TASK_ARRIVAL_RATE, display=False):
		super(Decision, self).__init__(time, "Decision")
		self.alg = algorithm
		self.nL = nL
		self.nodes = nodes
		self.ti = time_interval
		self.ar = ar

	def __str__(self):
		return "Decision["+("%.2f"%self.time)+"]["+self.nL.name+"]"

	def execute(self, eq):
		# decide for current node with incoming tasks
		if len(self.nL.w) > 0:
			Qsizes = []
			for n in self.nodes: Qsizes.append(n.qs())
			# state = (nL, w, Qsizes)
			state = (self.nL, len(self.nL.w), Qsizes)
			# algorithm decision
			if self.alg == "rd":  (w0, nO) = basic.randomalgorithm(state, self.nodes)
			if self.alg == "lq":  (w0, nO) = basic.leastqueue(state, self.nodes)
			if self.alg == "nn":  (w0, nO) = basic.nearestnode(state, self.nodes)

			# and execute the decision
			for i in range(w0):
				self.nL.send(self.nL.decide(), nO)
			for j in range(len(self.nL.w)):
				if not self.nL.fullqueue(): self.nL.queue(self.nL.decide())

			# if there is something to send, start the event
			if not self.nL.transmitting and self.nL.tosend():
				ev = Sending(self.time, self.nL)
				eq.addEvent(ev)

		# what to do with the rest of the w?

		# and add another decision after a time interval
		ev = Decision(self.time + self.ti, self.nL, self.nodes, self.alg, self.ti, self.ar)
		eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] [%.2f Decision]" % self.time)


# -------------------------------------------- Recieving --------------------------------------------

class Recieving(Event):
	def __init__(self, time, recieving_node, incoming_task=None, sending_node=None, ar=None,
		interval=configs.TIME_INTERVAL, nodes=None):
		super(Recieving, self).__init__(time, "Recieving")
		self.rn = recieving_node
		self.it = incoming_task
		if incoming_task == None: self.it = coms.Task(time)

		# the node that offloaded here
		self.sn = sending_node
		# or average arrival rate and interval in which we are considering
		self.ar = ar
		self.interval = interval
		self.nodes = nodes

	def __str__(self):
		retval = "Recieving["+("%.2f"%self.time)+"]["+self.rn.name+"]"
		return retval

	# recieves the tasks either from a client or from an offloading node
	def execute(self, eq):
		# if it comes from another offloading just try to queue it
		if self.sn is not None:
			t = self.rn.queue(self.it)
			self.sn.transmitting = False
		# else just recieve it for this time step
		else:
			t = self.rn.recieve(self.it)

		# start processing if it hasn't started already
		if not self.rn.processing and not self.rn.emptyqueue():
			ev = Processing(self.time, self.rn)
			eq.addEvent(ev)

		# start sending if there is stuff to send
		if not self.rn.transmitting and self.rn.tosend() > 0:
			ev = Sending(self.time, self.rn)
			eq.addEvent(ev)


		# and schedule the next event for recieving (poisson process)
		if self.ar is not None:
			ev = Recieving(self.time+utils.poissonNextEvent(self.ar, self.interval), utils.randomChoice(self.nodes),
				ar=self.ar, interval=self.interval, nodes=self.nodes)
			eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] [%.2f Recieving]" % self.time, "node", self.rn.name,
			"task timestamp %.2f" % self.it.timestamp, "task discarded", t)

		return t


# -------------------------------------------- Sending --------------------------------------------

class Sending(Event):
	def __init__(self, time, sending_node):
		super(Sending, self).__init__(time, "Sending")
		self.sn = sending_node

	def __str__(self):
		return "Sending["+("%.2f"%self.time)+"]["+self.sn.name+"]"

	# sends the task to another node, blocking coms in the meantime
	def execute(self, eq):
		# take from the coms queue a task
		[t, rn] = self.sn.popsendq()
		# if it cannot transmit, it fails
		if self.sn.comtime[rn] == -1: return t
		# then get busy
		self.sn.transmitting = True
		# recieves after comtime finished
		ev = Recieving(self.time+self.sn.comtime[rn], rn, t, self.sn)
		eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] [%.2f Sending]" % self.time, "from", self.sn.name,"to", rn.name)


# -------------------------------------------- Processing --------------------------------------------

class Processing(Event):
	def __init__(self, time, processing_node):
		super(Processing, self).__init__(time, "Processing")
		self.pn = processing_node

	def __str__(self):
		return "Processing["+("%.2f"%self.time)+"]["+self.pn.name+"][qs:"+str(self.pn.qs())+"]"

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
			# if the delay is negative break the code.
			if t.delay < 0: sys.exit()

		# debug message
		if configs.FOG_DEBUG == 1 and self.pn.processing:
			tstring = ("task timestamp %.2f" % t.timestamp)  + (" and delay %.2f" % t.delay)
			print("[DEBUG] [%.2f Processing]" % (self.time), "node", self.pn.name, tstring)

		return t
		
		