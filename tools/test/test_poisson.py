import sys
sys.path.append("/home/yourself/fog-computing-orchestration")

from tools import utils

def test_poisson():
	utils.initRandom()

	# average 5 events each 10 seconds
	pdist = utils.distOfWaitingTime(5, 10)

	# it should have each mili second
	assert len(pdist) == 10*1000

	assert round(pdist[len(pdist)-1]) == 1

	# for 1000 of simulation time
	t=0
	counts = []
	for i in range(0, 1000):
		counts.append(0)
	while t < 10000:
		t += utils.poissonNextEvent(5, 10)
		if t >= 10000: break
		counts[int(t/10)] += 1


	assert round(utils.listavg(counts)) == 5

