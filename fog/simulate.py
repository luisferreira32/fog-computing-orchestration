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
	# 1. check a state and make decisions
	# 2. generate round of recieving based on the new 'w' and repeat from 2.
	# 3. run events that generate more events

	# -------------------------------------------- 0. --------------------------------------------
	
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

	# create the event queue and cheat to start the loop
	evq = events.EventQueue()
	evq.q.append(events.Event(configs.SIM_TIME+1))
	ev = None
	check = 0

	# -------------------------------------------- 1. 2. 3. --------------------------------------------
	while evq.hasEvents():

		# -------------------------------------------- 1. --------------------------------------------

		# Make decision at every interval
		if ev is None or int(ev.time/configs.TIME_INTERVAL) >= check:
			check += 1
			print("CHECK",check)
			decision = {"w0" : 2, "n0" : nodes[1]}
			# don't forget to count the 'w' and the 'wL'

		# -------------------------------------------- 2. --------------------------------------------

		# Only need to add new tasks when there are none already
		if evq.recieving_client == 0:
			# if there is no event, this one was the first, else, we processed a recieving, add after interval
			if ev is None: clock = 0
			else: clock = ev.time + utils.poissonNextEvent(ar, configs.TIME_INTERVAL)
			newev = events.Recieving(clock, nodes[0], decision=decision, client=True)
			evq.addEvent(newev)
				

		# -------------------------------------------- 3. --------------------------------------------

		# pop and execute the first event that should be executed
		ev = evq.popEvent()
		t = ev.execute(evq)
		print(t)

	if configs.FOG_DEBUG == 1: print("[DEBUG] Finished simulation")
		