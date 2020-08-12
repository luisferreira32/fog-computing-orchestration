# external imports
import random
import queue

# local imports
from fog import node
from fog import configs
from fog import coms
from utils import graphs
from utils import utils
from algorithms import basic

def Simulate(sim_time=configs.SIM_TIME, n_nodes=configs.N_NODES, area=configs.MAX_AREA, influx=configs.TASK_ARRIVAL_RATE, sr=configs.SERVICE_RATE, algorithm=None,
	algorithm_object=None,
	debug_fog=False, debug_sim=False, display_q=False, display_wl=False, display_sum=False, display_w=False):
	"""Simulates a whole set up in a fog computing environment, given an algorithm
	
	Parameters
	----------
	sim_time
		the total number of seconds in the simulation
	n_nodes
		the number of nodes to generate
	area
		area configurations where to place the nodes
	influx
		the arrival rate of tasks in the nodes
	algorithm
		the offload algorithm for optimizing
	debug_fog, debug_sim
		flags to obtain debug messages
	display_q, display_wl, display_sum, display_w
		flags to display simulation graphs
	"""

	# We need a coreography algorithm somewhere....
	if algorithm==None:
		print("[SIM DEBUG] You need a valid algorithm to simulate")
		return None
	# and a seed reset to reproduce results
	random.seed(1)

	# ------------------------------------------------------------ SET UP ------------------------------------------------------------
	# N randomly placed nodes
	configs.FOG_DEBUG = debug_fog
	cps = sr*configs.DEFAULT_IL*configs.DEFAULT_CPI
	nodes = []
	for x in range(1,n_nodes+1):
		n1 = node.Core("n"+str(x), placement=(random.random()*area[0], random.random()*area[1]), influx=0, cpu=(configs.DEFAULT_CPI, cps))
		nodes.append(n1)

	rates12 = {}
	for node1 in nodes:
		for node2 in nodes:
			if node1 != node2:
				r12 = coms.transmissionrate(node1, node2)
				rates12[node1.name, node2.name] = r12 #
				#print("Comtime %.2f at distance %.2f" % (coms.comtime(1, r12), node.distance(node1, node2)))

	#print(rates12)

	# to keep track of the tasks offloaded and recieved
	recieving = {}
	offload = {}
	for x in nodes:
		for y in range(0,sim_time):
			recieving[x.name, y] = queue.Queue(100)
			offload[x.name, y] = 0

	# distribution of probabilities
	pdistribution = utils.poissondist(influx)

	# ---------------------------------------------------------- SIMULATION ----------------------------------------------------------
	worldclock = 0 # [s]
	configs.FOG_DEBUG = debug_fog

	# -- JUST FOR GRAPHS SAKE
	queues = {}
	wLs = {}
	ws = {}
	alldelays = []
	xclock = []
	totaldiscarded = 0
	# --

	# only one node recieving
	#nodes[0].setinflux(influx)

	# ------------------------------------------- SIMULATION MAIN LOOP ----------------------------------------------
	while worldclock < sim_time:
		if debug_sim: print("-------------------- second",worldclock,"-",worldclock+1,"--------------------")

		# ---------------- in each time step, the task arrival rate is a poisson distribution ----------------
		#for n in nodes:
		x = random.random()
		nodes[0].setinflux(utils.discreteX(pdistribution, x))

		# -- JUST FOR GRAPHS SAKE
		xclock.append(worldclock)
		for n in nodes:
			utils.appendict(queues, n.name, n.cpuqueue.qsize())
			utils.appendict(wLs, n.name, n.wL)
			utils.appendict(ws, n.name, n.influx)
		# --

		# ------------------------------------------ THIS IS WHERE THE ALGORITHM RUNS ----------------------------------------
		# Where it appends actions to the action bar [origin, dest, number_offloaded], given a state [current node, influx, Queue] and possible actions
		actions = []		
		for n in nodes:
			# -- run the algorithm only for nodes which have tasks allocated by the user! --
			if n.influx == 0:
				continue
			if algorithm=="lq": act = basic.leastqueue(n, nodes, recieving[n.name, worldclock])
			if algorithm=="rd": act = basic.randomalgorithm(n, nodes, recieving[n.name, worldclock])
			if algorithm=="nn": act = basic.nearestnode(n, nodes, recieving[n.name, worldclock])
			actions.extend(act)

		# ---------------------------------- Execute the offloading actions for every node -------------------------------------
		for (origin, dest, w0) in actions:
			# check the number of tasks offloaded on THIS node
			offload[origin.name, worldclock] += w0
			# it will always arrive in the next timestep, at least, then comtime is the roundtrip, so arrives in half time
			arriving_time = worldclock+1+int(coms.comtime(w0, rates12[origin.name, dest.name]))
			if arriving_time >= sim_time: # if they arrive after sim end, they won't be taken into account for this sim
				continue

			if debug_sim: print("[SIM DEBUG]",origin.name,"offloaded",w0,"tasks to node",dest.name,"arriving at",arriving_time)
			for x in range(0,w0):
				recieving[dest.name, arriving_time].put(origin.clock, False)

		# ------------------- Treat the tasks on the nodes, by queueing them and processing what can be processed -------------------
		# for every node run the decisions

		for n in nodes:
			# then process with the seconds we've got remaining, i.e. the second that's elapsing, plus the delay that the node clock has
			delays = n.process(1+(worldclock-n.clock))
			# -- JUST FOR GRAPHS SAKE
			alldelays.extend(delays)
			# --
			if debug_sim: print("[SIM DEBUG] Node",n.name,"clock at %.2f with task completion delays at"% n.clock,delays)

			# lastly decide which ones we'll work on and queue them
			totaldiscarded += n.queue(recieved=recieving[n.name, worldclock],offloaded = offload[n.name, worldclock])

		# end of the second
		worldclock +=1


	# --------------------------------------------- Print all the graphs and stats -----------------------------------------------
	# -- JUST FOR GRAPHS SAKE
	sumary = "Using algorithm " + str(algorithm) + " with SR " + str(sr) + " and average influx " + str(influx)
	if display_q: graphs.graphtime(xclock, queues, ylabel="queues", title=sumary)
	if display_wl: graphs.graphtime(xclock, wLs, ylabel="wLs")
	if display_w: graphs.graphtime(xclock, ws, ylabel="influx")
	if display_sum:
		print(sumary)
		print("  Avg delay is", utils.listavg(alldelays))
		print("  Total processed is", len(alldelays))
		print("  Total discarded is", totaldiscarded)
	return (utils.listavg(alldelays), len(alldelays), totaldiscarded, algorithm_object)
	# --
