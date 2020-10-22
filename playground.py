from fog.simulate import simulate
from fog import configs
from tools import utils, graphs
from algorithms import basic
import sys

algs = [basic.RandomAlgorithm(), basic.LeastQueueAlgorithm()]
algs = [basic.RandomAlgorithm()]
#algs = []
srs = [1, 3, 5, 7]
ars = [3, 5, 7, 9]
#placements= [(0,0), (10,10), (100,15), (95,85), (100,100)]
placements=None

#### --------------- ONE TIME SIMS 

if len(sys.argv) > 1 and sys.argv[1] == "test":
	configs.FOG_DEBUG = True
	configs.SIM_TIME = 0
	simulate(sr=1, placements=placements)
	sys.exit()


#### ------------------ GRAPHICAL SIMS
configs.SIM_TIME = 2000 # don't go over 50 000
configs.DEFAULT_DATA = 5*8
configs.INFO = 0


avg_reward_sr = {}
avg_delay_sr = {}
overload_sr = {}
for sr in srs:
	for alg in algs:
		(avg_reward, avg_delay, overload_prob)= simulate(sr=sr, algorithm_object=alg, placements=placements)
		utils.appendict(avg_reward_sr, alg, avg_reward)
		utils.appendict(avg_delay_sr, alg, avg_delay)
		utils.appendict(overload_sr, alg, overload_prob)
		print("[SR",sr,"]",alg)
		

avg_reward_ar = {}
avg_delay_ar = {}
overload_ar = {}
for ar in ars:
	for alg in algs:
		(avg_reward, avg_delay, overload_prob)= simulate(ar=ar, algorithm_object=alg, placements=placements)
		utils.appendict(avg_reward_ar, alg, avg_reward)
		utils.appendict(avg_delay_ar, alg, avg_delay)
		utils.appendict(overload_ar, alg, overload_prob)
		print("[AR",ar, "]",alg)

# graph time
xtimes = []
xlabels = []
for i in range(3):
	xtimes.append(srs)
	xtimes.append(ars)
	xlabels.append("srs")
	xlabels.append("ars")
liness = [avg_reward_sr, avg_reward_ar, avg_delay_sr, avg_delay_ar, overload_sr, overload_ar]
ylabels = [" "," ","s","s","%","%"]
titles = ["Average reward SR", "Average reward AR", "Average delay SR", "Average delay AR", "Overload prob SR", "Overload prob AR"]
graphs.all_graphtime(xtimes, liness, xlabels, ylabels, titles)
