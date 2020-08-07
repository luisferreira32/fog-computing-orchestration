# external imports
import random
random.seed(1)

# local imports
from fog import node
from fog import configs
from fog import coms
from utils import graphs
from utils import utils

# ------------------------------------------------------------ SET UP ------------------------------------------------------------
configs.FOG_DEBUG = 1
# five randomly placed nodes
nodes = []
for x in range(1,6):
	n1 = node.Core("n"+str(x), placement=(random.random()*configs.MAX_AREA[0], random.random()*configs.MAX_AREA[1]), influx=0)
	nodes.append(n1)

configs.FOG_DEBUG = 0
# calculate com ratios between them in the beginning @NOTE: if distance is bigger than 10m, it might not even be able to transmit!
rates12 = {}
for node1 in nodes:
	for node2 in nodes:
		if node1 != node2:
			r12 = coms.transmissionrate(node1, node2)
			rates12[node1.name, node2.name] = r12

print(rates12)


# ---------------------------------------------------------- SIMULATION ----------------------------------------------------------
# ------------------------------------------- SIMULATION SET UP ----------------------------------------------
worldclock = 0 # [s]
configs.FOG_DEBUG = 0
SIM_DEBUG = 1

# simulate for n iterations, focused on node 1 that's recieving tasks
nodes[0].addinflux(5)
nodes[1].addinflux(5)
nodes[2].addinflux(5)

# -- JUST FOR GRAPHS SAKE
queues = {}
wLs = {}
xclock = []
# --

# to keep track of the tasks offloaded and recieved
recieving = {}
offload = {}
for x in nodes:
	for y in range(0,configs.SIM_TIME):
		recieving[x.name, y] = 0
		offload[x.name, y] = 0

# ------------------------------------------- SIMULATION MAIN LOOP ----------------------------------------------
while worldclock < configs.SIM_TIME:
	print("-------------------- second",worldclock,"-",worldclock+1,"--------------------")

	# -- JUST FOR GRAPHS SAKE
	xclock.append(worldclock)
	# --
	
	# ------------------------------------------ THIS IS WHERE THE ALGORITHM RUNS ----------------------------------------
	# for every node takes in the recieving and offloading dicts, and adds tasks to them

	for node in nodes:
		# first make offloading decisions depending on state
		# -- JUST FOR GRAPHS SAKE
		utils.appendict(queues, node.name, node.cpuqueue)
		utils.appendict(wLs, node.name, node.wL)
		# --
		

		# -- tryout with random algorithm (start) --
		if node.excessinflux() > 0:
			# when i offload it's on this world clock time
			offload[node.name, worldclock] = node.excessinflux()
			e = offload[node.name, worldclock]

			# randomly offload the excess
			while e > 0:
				# choose a different node
				randomoff = node
				while randomoff == node:
					randomoff = random.choice(nodes)
				# and a random quantity to offload to that node
				er = int(random.random()*e)+1
				e -= er
				if SIM_DEBUG: print("[SIM DEBUG]",node.name,"offloaded",er,"tasks to node",randomoff.name)
				# but when i offload each task arrives at a different timestep
				while er > 0:
					arriving_time = worldclock+1 + int(coms.comtime(er, rates12[node.name, randomoff.name]))
					if arriving_time > configs.SIM_TIME:
						er -= 1 # BUGGED --- TASKS DISAPPEAR WITH HUGE DELAY THAT NO ONE'S IS ACCOUNTING FOR
						continue
					recieving[randomoff.name, arriving_time] += 1
					er -= 1				
		# -- tryout with random algorithm (end) --

	# ------------------- Treat the tasks on the nodes, by queueing them and processing what can be processed -------------------
	# for every node run the decisions

	for node in nodes:
		# then process with the seconds we've got remaining, i.e. the second that's elapsing, plus the delay that the node clock has
		node.process(1+(worldclock-node.clock))
		if SIM_DEBUG: print("[SIM DEBUG] Node",node.name,"with queue",node.cpuqueue,"has %.2f seconds behind world clock" % float(1+(worldclock-node.clock)))

		# lastly decide which ones we'll work on and queue them
		discard = node.setwL(recieved=recieving[node.name, worldclock],offloaded = offload[node.name, worldclock])
		if SIM_DEBUG: print("[SIM DEBUG] Node",node.name,"discarded",discard,"tasks.")
		node.queue()

	# end of the second
	worldclock +=1

# -- JUST FOR GRAPHS SAKE
#graphs.graphtime(xclock, queues, ylabel="queues")
#graphs.graphtime(xclock, wLs, ylabel="wLs")
# --