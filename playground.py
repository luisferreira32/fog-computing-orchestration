# external imports
import random
random.seed(1)

# local imports
from fog import node
from fog import configs
from fog import coms
from utils import graphs

# ------------------------------------------------------------ SET UP ------------------------------------------------------------
configs.FOG_DEBUG = 1
# five randomly placed nodes
nodes = []
for x in range(1,6):
	n1 = node.Core("n"+str(x), placement=(random.random()*configs.MAX_AREA[0], random.random()*configs.MAX_AREA[1]), influx=0)
	nodes.append(n1)

configs.FOG_DEBUG = 0
# calculate com ratios between them in the beginning @NOTE: if distance is bigger than 10m, it might not even be able to transmit!
rates12 = []
for node1 in nodes:
	rate12 = []
	for node2 in nodes:
		if node1 != node2:
			r12 = coms.transmissionrate(node1, node2)
			rate12.append(r12)
	rates12.append(rate12)
print(rates12)


# ---------------------------------------------------------- SIMULATION ----------------------------------------------------------
worldclock = 0 # [s]
configs.FOG_DEBUG = 1

# simulate for n iterations, focused on node 1 that's recieving tasks
nodes[0].addinflux(configs.TASK_ARRIVAL_RATE)

# -- JUST FOR GRAPHS SAKE
queues = []
xclock = []
for node in nodes:
	queue = []
	queues.append(queue)
n=0
# --

while worldclock < 10:
	print("-------------------- second",worldclock,"-",worldclock+1,"--------------------")

	# -- JUST FOR GRAPHS SAKE
	xclock.append(worldclock)
	n=0
	# --

	# to keep track of the tasks offloaded and recieved
	recieving = {}
	offload = {}
	for x in nodes:
		recieving[x.name] = 0
		offload[x.name] = 0
	
	# for every node make decisions
	for node in nodes:
		# first make offloading decisions depending on state
		# -- JUST FOR GRAPHS SAKE
		queues[n].append(node.cpuqueue)
		n +=1
		# --

		# ------------------------------------------ THIS IS THE RANDOM ALGORITHM ----------------------------------------
		if node.excessinflux() > 0:
			offload[node.name] = node.excessinflux()
			e = offload[node.name]

			# randomly offload the excess
			while e > 0:
				# choose a difer
				randomoff = node
				while randomoff == node:
					randomoff = random.choice(nodes)
				er = int(random.random()*e)+1
				e -= er
				recieving[randomoff.name] += er
				print("[DEBUG]",node.name,"offloaded",er,"tasks to node",randomoff.name)


	# ------------------- offloaded and recieved are the expected dicts after algorithm decision -------------------

	# for every node run the decisions
	for node in nodes:
		# then process with the seconds we've got remaining, i.e. the second that's elapsing, plus the delay that the node clock has
		node.process(1+(worldclock-node.clock))
		print("[DEBUG] Node",node.name,"has %.2f behind world clock" % float(1+(worldclock-node.clock)))
		# lastly decide which ones we'll work on and queue them
		node.setwL(recieved=recieving[node.name],offloaded = offload[node.name])
		node.queue()

	# end of the second
	worldclock +=1

# -- JUST FOR GRAPHS SAKE
graphs.graphqueues(xclock, queues)
# --