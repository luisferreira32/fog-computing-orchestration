# external imports
import collections

# our imports
from . import configs
from . import coms
from tools import utils
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
	def __init__(self, time, nodes, algorithm="rd", time_interval = configs.TIME_INTERVAL, 
		ar=configs.TASK_ARRIVAL_RATE, display=False):
		super(Decision, self).__init__(time, "Decision")
		self.alg = algorithm
		self.nodes = nodes
		self.ti = time_interval
		self.ar = ar
		self.display = display

	def execute(self, eq):
		# make the decision based on the current state of each node (checked by looking at nodes and edges)
		new_decisions = {}
		# generate decision for every client connected node
		for nL in self.nodes:
			if nL.w == 0: continue
			Qsizes = []
			for n in self.nodes: Qsizes.append(n.qs())
			# state = (nL, w, Qsizes)
			state = (nL, nL.w, Qsizes)
			# algorithm decision
			if self.alg == "rd":  (w0, nO_index) = basic.randomalgorithm(state)
			if self.alg == "lq":  (w0, nO_index) = basic.leastqueue(state)
			if self.alg == "nn":  (w0, nO_index) = basic.nearestnode(state)

			# and wrap the new decision
			new_decisions[nL] = {"w0": w0, "nO": self.nodes[nO_index]}
			nL.w = 0

		# and for every Recieving event in the evq, change it's decision to the new one
		for ev in eq.q:
			if ev.classtype == "Recieving" and ev.decision is not None:
				for n in self.nodes:
					if ev.rn == n: ev.decision = new_decisions[n]

		# and add another decision after a time interval
		ev = Decision(self.time + self.ti, self.nodes, self.alg, self.ti, self.ar)
		eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] Executed decision at %0.2f" % self.time)

		# if we're going to display the messages until now


# -------------------------------------------- Recieving --------------------------------------------

class Recieving(Event):
	def __init__(self, time, recieving_node, incoming_task=None, decision=None, sending_node=None, 
		ar=None, interval=None):
		super(Recieving, self).__init__(time, "Recieving")
		self.rn = recieving_node
		self.e = recieving_node.edges
		self.it = incoming_task
		if incoming_task == None: self.it = coms.Task(time)

		# a set for the decision for the next "w" tasks on this node: [nO, w0] with w0 < w
		self.decision = decision
		# or the node that offloaded here
		self.sn = sending_node
		# average arrival rate and interval in which we are considering
		self.ar = ar
		self.interval = interval

	# allocs a task to node queue, offloads to another or discards.
	def execute(self, eq):
		# if it comes from another offloading just try to queue it
		if self.decision is None and self.sn is not None:
			self.sn.edges[self.rn].busy = False
			t = self.rn.queue(self.it)
		# if we're meant to offload and we can... do it
		elif self.decision["w0"] > 0 and not self.e[self.decision["nO"]].busy:
			self.decision["w0"] -= 1
			ev = Sending(self.time, self.rn, self.decision["nO"], self.it)
			eq.addEvent(ev)
			t = None
			self.rn.w += 1
		else:
			# queue the task if we didn't offload it, returns task if queue is full
			t = self.rn.queue(self.it)
			self.rn.w += 1

		# start processing if it hasn't started already
		if not self.rn.processing:
			ev = Processing(self.time, self.rn)
			eq.addEvent(ev)

		# and schedule the next event for recieving (poisson process)
		if self.ar is not None:
			ev = Recieving(self.time+utils.poissonNextEvent(self.ar, self.interval), self.rn,
				decision=self.decision, ar=self.ar, interval=self.interval)
			eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] Executed recieving at %0.2f" % self.time)

		return t


# -------------------------------------------- Sending --------------------------------------------

class Sending(Event):
	def __init__(self, time, sending_node, recieving_node, outbound_task):
		super(Sending, self).__init__(time, "Sending")
		self.sn = sending_node
		self.rn = recieving_node
		self.ot = outbound_task
		self.edge = self.sn.edges[self.rn]

	# sends the task to another node, blocking coms in the meantime
	def execute(self, eq):
		self.edge.busy = True
		# if there is no connection, it fails
		if self.edge.comtime == -1: return outbound_task
		# recieves after comm time finished
		ev = Recieving(self.time+self.edge.comtime, self.rn, incoming_task=self.ot, sending_node=self.sn)
		eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] Executed sending at %0.2f" % self.time)


# -------------------------------------------- Processing --------------------------------------------

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

		# debug message
		if configs.FOG_DEBUG == 1 and self.pn.processing:
			print("[DEBUG] Executed processing at %0.2f" % self.time)

		return t		
		
		