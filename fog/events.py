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
	def __init__(self, time, nodes, algorithm="rd", time_interval = configs.TIME_INTERVAL, 
		ar=configs.TASK_ARRIVAL_RATE, display=False):
		super(Decision, self).__init__(time, "Decision")
		self.alg = algorithm
		self.nodes = nodes
		self.ti = time_interval
		self.ar = ar
		self.display = display

	def __str__(self):
		return "Decision["+("%.2f"%self.time)+"]"

	def execute(self, eq):
		# make the decision based on the current state of each node (checked by looking at nodes and edges)
		new_decisions = {}
		# generate decision for every client connected node
		for nL in self.nodes:
			#if nL.w == 0: continue
			Qsizes = []
			for n in self.nodes: Qsizes.append(n.qs())
			# state = (nL, w, Qsizes)
			state = (nL, self.ar, Qsizes)
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
					if ev.rn == n: 
						ev.decision = new_decisions[n]

		# and add another decision after a time interval
		ev = Decision(self.time + self.ti, self.nodes, self.alg, self.ti, self.ar)
		eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] [%.2f Decision]" % self.time)
		if configs.FOG_DEBUG == 1: graphs.displayState(self.time,self.nodes, new_decisions, eq)

		# if we're going to display the messages until now


# -------------------------------------------- Recieving --------------------------------------------

class Recieving(Event):
	def __init__(self, time, recieving_node, incoming_task=None, decision=None, sending_node=None, 
		ar=None, interval=None, nodes=None):
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
		self.nodes = nodes

	def __str__(self):
		retval = "Recieving["+("%.2f"%self.time)+"]["+self.rn.name+"]"
		if self.decision and self.decision["w0"] > 0:
			retval += "{w0:"+str(self.decision["w0"])+" nO:"+self.decision["nO"].name+"}"
		else:
			retval += "{"+"-----}"
		retval += "[t:"+("%.2f"%self.it.timestamp)+"]"
		return retval

	# allocs a task to node queue, offloads to another or discards.
	def execute(self, eq):
		# if it comes from another offloading just try to queue it
		if self.decision is None and self.sn is not None:
			self.sn.edges[self.rn].busy = False
			t = self.rn.queue(self.it)
		# if we're meant to offload try to do it #and self.decision["nO"] != self.rn 
		elif self.decision["w0"] > 0 and not self.rn.edges[self.decision["nO"]].busy:
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
				decision=self.decision, ar=self.ar, interval=self.interval, nodes=self.nodes)
			eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] [%.2f Recieving]" % self.time, "node", self.rn.name,
			"task timestamp %.2f" % self.it.timestamp, "task discarded", t)

		return t


# -------------------------------------------- Sending --------------------------------------------

class Sending(Event):
	def __init__(self, time, sending_node, recieving_node, outbound_task):
		super(Sending, self).__init__(time, "Sending")
		self.sn = sending_node
		self.rn = recieving_node
		self.ot = outbound_task
		self.edge = sending_node.edges[recieving_node]

	def __str__(self):
		retval = "Sending["+("%.2f"%self.time)+"]["+self.sn.name+"->"+self.rn.name+"]"
		retval += "[t:"+("%.2f"%self.ot.timestamp)+"]"
		return retval

	# sends the task to another node, blocking coms in the meantime
	def execute(self, eq):
		# if there is no connection, it fails
		if self.edge.comtime == -1: return self.ot
		# then get busy
		self.edge.busy = True
		# recieves after comm time finished
		ev = Recieving(self.time+self.edge.comtime, self.rn, self.ot, None, self.sn)
		eq.addEvent(ev)

		# debug message
		if configs.FOG_DEBUG == 1: print("[DEBUG] [%.2f Sending]" % self.time, "from", self.sn.name,"to",
			self.rn.name,"task timestamp %.2f" % self.ot.timestamp, "arriving %.2f" % (self.time+self.edge.comtime))


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
		
		