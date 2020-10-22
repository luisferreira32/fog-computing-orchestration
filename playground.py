# fog packages
from fog.simulate import simulate
from fog.envrionment import FogEnv
from fog import configs

# tools
from tools import utils, graphs
import sys

#algorithms
from algorithms import basic
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C

# --------------------------------------------THE PLAYGROUND-----------------------------------------

# variables for graphical display
avg_reward_sr = {}; avg_delay_sr = {}; overload_sr = {}; 
avg_reward_ar = {}; avg_delay_ar = {}; overload_ar = {};

# variables for the simulation configuration
srs = [1, 3, 5, 7]
ars = [3, 5, 7, 9]
configs.SIM_TIME = 2000 # don't go over 50 000
configs.DEFAULT_DATA = 5*8
configs.INFO = 0

#### ------------------ SETU UP THE ENV ------------------



#### ----------------- SET UP ALGORITHMS -----------------
# set up which algorithms will run 
algs = []
algs.append(basic.RandomAlgorithm())
#algs.append(basic.LeastQueueAlgorithm())
algs.append(PPO2(MlpPolicy, FogEnv(None), verbose=1))
algs.append(A2C(MlpPolicy, FogEnv(None), verbose=1)	)

#### ------------------ GRAPHICAL SIMS ------------------

# simulations
for sr in srs:
	for alg in algs:
		(avg_reward, avg_delay, overload_prob)= simulate(sr=sr, algorithm=alg)
		utils.appendict(avg_reward_sr, alg, avg_reward)
		utils.appendict(avg_delay_sr, alg, avg_delay)
		utils.appendict(overload_sr, alg, overload_prob)
		print("[SR",sr,"]",alg)
		

for ar in ars:
	for alg in algs:
		(avg_reward, avg_delay, overload_prob)= simulate(ar=ar, algorithm=alg)
		utils.appendict(avg_reward_ar, alg, avg_reward)
		utils.appendict(avg_delay_ar, alg, avg_delay)
		utils.appendict(overload_ar, alg, overload_prob)
		print("[AR",ar, "]",alg)

# -------------------- print them all --------------------

xtimes = []; xlabels = [];
for i in range(3):
	xtimes.append(srs)
	xtimes.append(ars)
	xlabels.append("srs")
	xlabels.append("ars")
liness = [avg_reward_sr, avg_reward_ar, avg_delay_sr, avg_delay_ar, overload_sr, overload_ar]
ylabels = [" "," ","s","s","%","%"]
titles = ["Average reward SR", "Average reward AR", "Average delay SR", "Average delay AR", "Overload % SR", "Overload % AR"]
graphs.all_graphtime(xtimes, liness, xlabels, ylabels, titles)
