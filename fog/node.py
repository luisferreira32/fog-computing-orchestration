# external imports
import math
import collections

# import necessary fog environment configurations
from . import configs
from . import coms

#------------------------------------------------------ ------------- -----------------------------------------------------
#--------------------------------------------------- Fog Node Core Class --------------------------------------------------
#------------------------------------------------------ ------------- -----------------------------------------------------

class Core(object):
	#The core of a fog computing node has all its attributes.
	def __init__(self, name="default_node", cpu=(configs.DEFAULT_CPI, configs.DEFAULT_CPS), placement=(0,0),
		 bandwidth=configs.DEFAULT_BANDWIDTH, power=configs.DEFAULT_POWER):
	
		# set up the attributes
		# generic
		self.name = name
		self.placement = placement
		# cpu related
		self.cpi = cpu[0]
		self.cps = cpu[1]
		self.cpuqueue = collections.deque(maxlen=configs.MAX_QUEUE)
		self.processing = False
		# tasks allocated in this node per unit time
		self.w = collections.deque(maxlen=configs.MAX_W)
		# communications
		self.bw = bandwidth
		self.pw = power
		self.comtime = {}
		self.transmitting = False
		self.sendq = collections.deque()

		# and debug if set to do so
		if configs.FOG_DEBUG:
			print("[DEBUG] Node core "+self.name+" created: "+str(self.__dict__))


	# --- CPU related methods --- 

	def emptyqueue(self):
		#Checks if CPU queue is empty
		return len(self.cpuqueue) == 0

	def fullqueue(self):
		#Checks if CPU queue is full
		return len(self.cpuqueue) == self.cpuqueue.maxlen

	def qs(self):
		#Checks CPU queue size
		return len(self.cpuqueue)

	def process(self, time):
		#Process the first task in the CPU queue and return the task, time is the current time

		# it's not supposed to be empty...
		if self.emptyqueue():
			return None

		# and it takes a while to process a task
		t = self.cpuqueue.popleft()
		time = time + t.il*self.cpi/self.cps
		t.process(time)
		return t

	def queue(self, t):
		#Add a task to the cpu queue and set the locally processed tasks number
		# queue is full, give the task back
		if self.fullqueue():
			return t
		# else add it to the queue and return None
		self.cpuqueue.append(t)
		return None

	# --- communication related methods ---

	def setcomtime(self, nodes):
		# sets the communication edges time with other nodes
		for n in nodes:
			if n == self: continue
			self.comtime[n] = coms.comtime(1, coms.transmissionrate(self, n, self.bw, self.pw))

	def recieve(self, t):
		# if W_MAX exceeded, discard task, else get it
		if len(self.w) == configs.MAX_W:
			return t
		self.w.append(t)
		return None

	def decide(self):
		if len(self.w) == 0:
			return None
		return self.w.popleft()

	def send(self, task, destination):
		self.sendq.append([task, destination])

	def popsendq(self):
		return self.sendq.popleft()

	def tosend(self):
		if len(self.sendq) > 0:
			return True
		return False


#------------------------------------------------------ ------------ -----------------------------------------------------
#--------------------------------------------------- Functions on nodes --------------------------------------------------
#------------------------------------------------------ ------------ -----------------------------------------------------

def avgcps(n=Core(), sr=configs.SERVICE_RATE):
	#Calculate cycles per second (*10^6) considering a model node, model task and a model service rate
	return sr*n.cpi*configs.DEFAULT_IL

def extime(n1=None, n2=None, wL=0, w0=0):
	#Calculate task execution time on node local and offloaded
	if n1 is None or n2 is None or w0 < 0 or wL < 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Invalid parameters in extime()")
		return -1

	return ((configs.DEFAULT_IL*n1.cpi*wL)/n1.cps + (configs.DEFAULT_IL*n2.cpi*w0)/n2.cps)


def wtime(n1=None, n2=None, wL=0, w0=0, sr=configs.SERVICE_RATE):
	#Calculate the average waiting time
	if n1 is None or n2 is None or w0 < 0 or wL < 0:
		if configs.FOG_DEBUG == 1: print("[DEBUG] Invalid parameters in wtime()")
		return -1

	# wt = (QL/srL)[if wL != 0] + (QL/srL + Q0/sr0)[if w0 != 0]
	wt = 0
	if wL > 0: wt += n1.qs()/sr
	if w0 > 0: wt += n1.qs()/sr + n2.qs()/sr
	return wt


def distance(n1=None, n2=None):
	#Calculate the distance between two nodes on a plane
	if n1 is None or n2 is None:
		if configs.FOG_DEBUG == 1: print("[DEBUG] None argument in distance()")
		return -1
	try:
		return math.sqrt((n1.placement[0] - n2.placement[0])**2 + (n1.placement[1] - n2.placement[1])**2)
	except Exception as InvalidParameters:
		raise InvalidParameters