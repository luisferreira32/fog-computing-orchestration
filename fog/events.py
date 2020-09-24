# external imports
import collections

# our imports
from . import configs
from . import coms
from tools import utils
from algorithms import basic


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



class Event(object):
	def __init__(self, time, classtype=None):
		self.time = time
		self.classtype = classtype



class Decision(Event):
	def __init__(self, time, nodes, algorithm="random", time_interval = configs.TIME_INTERVAL, 
		ar=configs.TASK_ARRIVAL_RATE):
		super(Decision, self).__init__(time, "Decision")
		self.alg = algorithm
		self.nodes = nodes
		self.ti = time_interval
		self.ar = ar

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
			if self.alg == "random":  (w0, nO_index) = basic.randomalgorithm(state)
			if self.alg == "leastqueue":  (w0, nO_index) = basic.leastqueue(state)
			if self.alg == "nearestnode":  (w0, nO_index) = basic.nearestnode(state)

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



class Recieving(Event):
	def __init__(self, time, recieving_node, edges=None, incoming_task=None, decision=None, 
		sending_node=None, client_dist=None):
		super(Recieving, self).__init__(time, "Recieving")
		self.rn = recieving_node
		self.e = edges
		self.it = incoming_task
		if incoming_task == None: self.it = coms.Task(time)

		# a set for the decision for the next "w" tasks on this node: [nO, w0] with w0 < w
		self.decision = decision
		# or the node that offloaded here
		self.sn = sending_node
		# mass distribution to set up next client event 
		self.client_dist = client_dist

	# allocs a task to node queue, offloads to another or discards.
	def execute(self, eq):
		# if it comes from another offloading just try to queue it
		if self.decision is None and self.sn is not None:
			self.e.busy = False
			t = self.rn.queue(self.it)
		# if we're meant to offload and we can... do it
		elif self.decision["w0"] > 0 and not self.e[self.decision["nO"]].busy:
			self.decision["w0"] -= 1
			ev = Sending(self.time, self.rn, self.decision["nO"], self.it, self.e[self.decision["nO"]])
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
		if self.client_dist is not None:
			ev = Recieving(self.time+utils.poissonNextEvent(self.client_dist), self.rn, self.e,
				decision=self.decision, client_dist=self.client_dist)
			eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] Executed recieving at %0.2f" % self.time)

		return t


class Sending(Event):
	def __init__(self, time, sending_node, recieving_node, outbound_task, edge):
		super(Sending, self).__init__(time, "Sending")
		self.sn = sending_node
		self.rn = recieving_node
		self.ot = outbound_task
		self.edge = edge

	# sends the task to another node, blocking coms in the meantime
	def execute(self, eq):
		self.edge.busy = True
		# recieves after comm time finished
		ev = Recieving(self.time+self.edge.comtime, self.rn, edges=self.edge,
			incoming_task=self.ot, sending_node=self.sn)
		eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] Executed sending at %0.2f" % self.time)



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
		if configs.FOG_DEBUG == 1: print("[DEBUG] Executed processing at %0.2f" % self.time)

		return t		
		
		