import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from tools import utils

def test_uniform_dist_with_poisson_arrival():
	utils.initRandom()

	t = 0
	final_time = 10000
	nodes = [0, 0, 0, 0, 0]
	# for a sim time of 10 000, in timestep=1 and lbd=5
	while t < final_time:
		t += utils.poissonNextEvent(5, 1)
		if t>=final_time: break
		nodes[int(utils.uniformRandom(5))] += 1

	# so the number of average tasks per node was 5/5 = 1
	for n in nodes:
		assert round(n/final_time, 1) == 1.0
	# and total average was 5	
	assert round(sum(nodes)/final_time,1) == 5.0