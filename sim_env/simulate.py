# fog related imports
from fog import configs, node
from fog.envrionment import CreatFogEnv
from stable_baselines.common.vec_env import DummyVecEnv
# tools
from tools import utils


def simulate(algorithm=None, env=None):

	# ------------------------------------------ set up the env ------------------------------------------
	if algorithm is None or env is None: return -1, -1, -1
	# initiate a constant random - simulation consistency
	utils.initRandom()
	env = DummyVecEnv([lambda: env]) 
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
		if rw != -1000.0: rewards.append(rw) # average reward of possible actions
		discarded += info["discarded"]
		env.render()

	return utils.listavg(rewards), utils.listavg(delays), round(discarded/(discarded+len(delays)),3)

		
def algorithm_classroom(srs, ars, algorithm=None):
	print("[INFO] training algorithm",algorithm)

	for sr in srs:
		print("[SR",sr,"]")
		# -- set up the envrionment depending on the sr and ar
		env = CreatFogEnv(sr, configs.TASK_ARRIVAL_RATE)
		env = DummyVecEnv([lambda: env])
		# --- and learn a bit of the new envrionment ---
		algorithm.set_env(env)
		algorithm.learn(total_timesteps=configs.SIM_TIME)

	for ar in ars:
		print("[AR",ar,"]")
		env = CreatFogEnv(configs.SERVICE_RATE, ar)
		env = DummyVecEnv([lambda: env])
		# --- and learn a bit of the new envrionment ---
		algorithm.set_env(env)
		algorithm.learn(total_timesteps=configs.SIM_TIME)