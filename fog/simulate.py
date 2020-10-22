# 'w' is defined here and passed as argument for the events
# IF AFTER EXECUTING a returning task is not completed, it was discarded 
# create a time dictionary of communication so it doesn't need to do math every time

# fog related imports
from . import configs
from . import node
from . import events
from . import envrionment

# tools
from tools import utils, graphs

# decision algorithms
from algorithms import basic

def simulate(sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE, algorithm_object=None, placements=None):

	# ------------------------------------------ create the nodes ------------------------------------------
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

	# ------------------------------------------ create the env ------------------------------------------
	env = envrionment.FogEnv(nodes, sr, ar)
	# and some vars
	delays = []; discarded = 0;

	# -------------------------------------------- run the loop ------------------------------------------
	obs = env.reset()
	for t in range(configs.SIM_TIME):
		action = algorithm_object.execute(obs)
		obs, rewards, done, info = env.step(action)
		delays.extend(info["delays"])
		discarded += info["discarded"]
		env.render()

	return 1, utils.listavg(delays), round(discarded/(discarded+len(delays)),3)

		
