# fog related imports
from fog import configs, envrionment, node
from stable_baselines.common.vec_env import DummyVecEnv
# tools
from tools import utils


def simulate(sr=configs.SERVICE_RATE, ar=configs.TASK_ARRIVAL_RATE, algorithm=None):

	# ------------------------------------------ set up the env ------------------------------------------
	if algorithm == None: return -1, -1, -1
	# initiate a constant random - simulation consistency
	utils.initRandom()
	# placement of the nodes
	placements=[]
	for i in range(0, configs.N_NODES):
		placements.append((utils.uniformRandom(configs.MAX_AREA[0]),utils.uniformRandom(configs.MAX_AREA[1])))

	# the nodes 
	cps = sr*configs.DEFAULT_IL*configs.DEFAULT_CPI/configs.TIME_INTERVAL
	nodes = []
	for i in range(0, configs.N_NODES):
		n = node.Core(name="n"+str(i), index=i,	placement=placements[i], cpu=(configs.DEFAULT_CPI, cps))
		nodes.append(n)
	# create M edges between each two nodes
	for n in nodes:
		n.setcomtime(nodes)

	env = envrionment.FogEnv(nodes, sr, ar)
	env = DummyVecEnv([lambda: env])
	# --- and learn a bit of the new envrionment ---
	algorithm.set_env(env)
	algorithm.learn(total_timesteps=configs.SIM_TIME*2)

	# -------------------------------------------- run the loop ------------------------------------------
	# some info registering apps
	rewards = []; delays = []; discarded = 0;
	obs = env.reset()
	for t in range(configs.SIM_TIME):
		action, _states = algorithm.predict(obs)
		obs, rw, done, info = env.step(action)
		# unpack and save it
		rw = rw[0]; info = info[0];
		delays.extend(info["delays"])
		rewards.append(rw)
		discarded += info["discarded"]
		env.render()

	return utils.listavg(rewards), utils.listavg(delays), round(discarded/(discarded+len(delays)),3)

		
