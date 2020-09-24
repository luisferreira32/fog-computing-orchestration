# 'w' is defined here and passed as argument for the events
# IF AFTER EXECUTING a returning task is not completed, it was discarded 
# create a time dictionary of communication so it doesn't need to do math every time

# fog related imports
from . import configs
from . import node
from . import events
from . import coms

# tools
from tools import utils

# decision algorithms
from algorithms import basic

def simulate(sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE, algorithm="rd"):
	# 0. create all necessary information for the simulation to begin
	# 1. create first round of events (decision and recieving tasks)
	# 2. run events that generate more events
	# 3. repeat from 2

	# -------------------------------------------- 0. --------------------------------------------
	
	# initiate a constant random - simulation consistency
	utils.initRandom()

	# create N_NODES with random placements within a limited area and a configured SR
	nodes = []
	# cycles per second, depends on the TIME INTERVAL of the SERVICE RATE
	cps = sr*configs.DEFAULT_IL*configs.DEFAULT_CPI/configs.TIME_INTERVAL
	for i in range(0, configs.N_NODES):
		n = node.Core(name="n"+str(i), 
			placement=(utils.uniformRandom(configs.MAX_AREA[0]),utils.uniformRandom(configs.MAX_AREA[1])),
			cpu=(configs.DEFAULT_CPI, cps))
		nodes.append(n)
	# create M edges between each two nodes
	for n in nodes:
		n.setedges(nodes)

	# create the event queue
	evq = events.EventQueue()

	# and for information obtaining
	delays = []

	# -------------------------------------------- 1. --------------------------------------------

	# begin the first client request, that calls another based on a poisson process
	pdist = utils.distOfWaitingTime(ar, configs.TIME_INTERVAL)
	ev = events.Recieving(0, nodes[0], decision={"w0":0, "n0":None}, client_dist=pdist)
	evq.addEvent(ev)
	ev = events.Recieving(0, nodes[1], decision={"w0":0, "n0":None}, client_dist=pdist)
	evq.addEvent(ev)
	# decision making time
	ev = events.Decision(0, nodes, algorithm, ar=ar)
	evq.addEvent(ev)

	# -------------------------------------------- 2. 3. --------------------------------------------
	while evq.hasEvents():
		ev = evq.popEvent()
		# -------------------------------------------- 2. --------------------------------------------
		
		# execute the first event of the queue
		t = ev.execute(evq)
		if t is not None:
			delays.append(t.delay)

		# -------------------------------------------- 3. --------------------------------------------

		# It's repeating until queue ends, which is the last event scheduled before simulation limit time

	if configs.FOG_DEBUG == 1: print("[DEBUG] Finished simulation")

	return utils.listavg(delays)
		