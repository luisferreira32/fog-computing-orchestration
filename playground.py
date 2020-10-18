from fog.simulate import simulate
from fog import configs
from tools import utils, graphs
from algorithms import basic,qlearning
import sys

algs = [basic.RandomAlgorithm(), basic.LeastQueueAlgorithm(), basic.NearestNodeAlgorithm()]
srs = [1, 3, 5, 7]
ars = [3, 5, 7, 9]
placements= [(0,0), (10,10), (100,15), (95,85), (100,100)]

#### --------------- ONE TIME SIMS 

if len(sys.argv) > 1 and sys.argv[1] == "display":
	configs.DISPLAY = True
	configs.SIM_TIME = 100
	ao = qlearning.Qlearning(sr=8, ar=5.2)
	simulate(sr=8, ar=5.2, algorithm_object=algs[0])
	configs.DISPLAY = False
	sys.exit()

if len(sys.argv) > 1 and sys.argv[1] == "test":
	configs.FOG_DEBUG = True
	configs.SIM_TIME = 0
	simulate(sr=1, placements=placements)
	sys.exit()


#### ------------------ GRAPHICAL SIMS
configs.SIM_TIME = 2000 # don't go over 50 000


# different Q with dif parameters
ao1 = qlearning.Qlearning(sr=1, ar=5.2)
algs.append(ao1)
ao2 = qlearning.Qlearning(sr=1, ar=5.2)
ao2.r_utility = 0
ao2.x_delay = 0
ao2.x_overload = 0
#algs.append(ao2)


avg_delay_sr = {}
for sr in srs:
	for alg in algs:
		if alg.updatable:
			alg.changeiter(epsilon=0.9,sr=sr, ar=5.2)
		(avg_delay, processed, discarded)= simulate(sr=sr, algorithm_object=alg, placements=placements)
		utils.appendict(avg_delay_sr, alg, avg_delay)
		print("[SR",sr,"]",alg, "total",processed+discarded, "overloaded", round(discarded/(processed+discarded),3))
		

avg_delay_ar = {}
for ar in ars:
	for alg in algs:
		if alg.updatable:
			alg.changeiter(epsilon=0.9,ar=ar, sr=1.8)
		(avg_delay, processed, discarded)= simulate(ar=ar, algorithm_object=alg, placements=placements)
		utils.appendict(avg_delay_ar, alg, avg_delay)
		print("[AR",ar, "]",alg ,"total",processed+discarded, "overloaded", round(discarded/(processed+discarded),3))

#print("Possible states",10*10*10*10*10*5*20, "states visited",len(ao1.qtable))
graphs.graphtime(srs, avg_delay_sr)
graphs.graphtime(ars, avg_delay_ar)
