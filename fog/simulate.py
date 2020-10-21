# 'w' is defined here and passed as argument for the events
# IF AFTER EXECUTING a returning task is not completed, it was discarded 
# create a time dictionary of communication so it doesn't need to do math every time

# fog related imports
from . import configs
from . import node
from . import events
from . import coms
from .controller import Controller

# tools
from tools import utils, graphs

# decision algorithms
from algorithms import basic

def simulate(sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE, algorithm_object=None, placements=None):
	# 0. create all necessary information for the simulation to begin
	# 1. create first round of events (decision and recieving tasks)
	# 2. run events that generate more events
	# 3. repeat from 2

	# -------------------------------------------- 0. --------------------------------------------
	
	# initiate a constant random - simulation consistency
	utils.initRandom()
	cps = sr*configs.DEFAULT_IL*configs.DEFAULT_CPI/configs.TIME_INTERVAL
	if placements is None:
		placements = []
		for i in range(0, configs.N_NODES):
			placements.append((utils.uniformRandom(configs.MAX_AREA[0]),utils.uniformRandom(configs.MAX_AREA[1])))

	# create N_NODES with random placements within a limited area and a configured SR
	nodes = []
	for i in range(0, configs.N_NODES):
		n = node.Core(name="n"+str(i), index=i,	placement=placements[i], cpu=(configs.DEFAULT_CPI, cps))
		nodes.append(n)
	# create M edges between each two nodes
	for n in nodes:
		n.setcomtime(nodes)

	# create the event queue
	evq = events.EventQueue()

	# and for information obtaining
	delays = []; discarded = 0; x=0;
	sends = 0; recieves = 0; processes = 0;

	# lastly the controller that'll run the algorithm
	if algorithm_object is None: algorithm_object = basic.RandomAlgorithm(nodes)
	algorithm_object.setnodes(nodes)
	ctr = Controller(nodes, algorithm_object)

	# -------------------------------------------- 1. --------------------------------------------

	# begin the first client request, that calls another based on a poisson process
	ev = events.Recieving(0, nodes[0], ar=ar, interval=configs.TIME_INTERVAL, nodes=nodes)
	evq.addEvent(ev)
	ev = events.Decision(0, ctr)
	evq.addEvent(ev)

	# -------------------------------------------- 2. 3. --------------------------------------------
	while evq.hasEvents():
		ev = evq.popEvent()
		# -------------------------------------------- 2. --------------------------------------------
		
		# execute the first event of the queue
		t = ev.execute(evq)
		if t is not None:
			if isinstance(t, int): discarded += t
			elif t.delay == -1: discarded += 1
			else: delays.append(t.delay)

		# -------------------------------------------- 3. --------------------------------------------

		# some info
		if ev.classtype == "Recieving": recieves +=1
		elif ev.classtype == "Sending": sends +=1
		elif ev.classtype == "Processing": processes +=1
		# just a loading display
		if ev.time == x:
			print(".", end="",flush=True)
			x += int(configs.SIM_TIME/5)

	if configs.FOG_DEBUG == 1: print("[DEBUG] Finished simulation")
	if configs.INFO == 1: print(" [INFO]",algorithm_object," Ran",sends,"SENDING")#,recieves,"RECIEVINGS",processes,"PROCESSING events")

	for n in nodes:
		discarded += len(n.w) + len(n.sendq)
	# return avg reward, avg delay, and overload probability
	return (utils.listavg(ctr.rewards), utils.listavg(delays), round(discarded/(len(delays)+discarded),3))
		
