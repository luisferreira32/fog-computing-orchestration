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

def simulate(sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE):
	# 0. create all necessary information for the simulation to begin
	# 1. check a state and make decisions within DECISION events
	# 2. run events that generate more events
	# 3. repeat from 1

	# -------------------------------------------- 0. --------------------------------------------
	
	# initiate a constant random - simulation consistency
	utils.initRandom()

	# create N_NODES with random placements within a limited area and a configured SR
	nodes = []
	# cycles per second, depends on the TIME INTERVAL of the SERVICE RATE
	cps = sr*configs.DEFAULT_IL*configs.DEFAULT_CPI/configs.TIME_INTERVAL
	for i in range(1, configs.N_NODES):
		n = node.Core(name="n"+str(i), 
			placement=(utils.uniformRandom(configs.MAX_AREA[0]),utils.uniformRandom(configs.MAX_AREA[1])),
			cpu=(configs.DEFAULT_CPI, cps))
		nodes.append(n)
	for n in nodes:
		n.setcomstime(nodes)

	# create the event queue
	evq = events.EventQueue()
	# begin the first client request, that calls another based on a poisson process
	pdist = utils.distOfWaitingTime(ar, configs.TIME_INTERVAL)
	ev = events.Recieving(0, nodes[0], decision={"w0":0, "n0":None}, client_dist=pdist)
	evq.addEvent(ev)
	# decision making time
	check = 0

	# -------------------------------------------- 1. 2. 3. --------------------------------------------
	while evq.hasEvents():
		ev = evq.popEvent()

		# -------------------------------------------- 1. --------------------------------------------


		# -------------------------------------------- 2. --------------------------------------------
		
		# execute the first event of the queue
		t = ev.execute(evq)
		if t is not None:
			print(t.delay)

		# -------------------------------------------- 3. --------------------------------------------

		# It's repeating until queue ends, which is the last event scheduled before simulation limit time

	if configs.FOG_DEBUG == 1: print("[DEBUG] Finished simulation")
		