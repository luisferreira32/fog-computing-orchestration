import sys
sys.path.append("/home/yourself/fog-computing-orchestration")

from tools import utils

def test_poisson():
	utils.initRandom()

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

		

