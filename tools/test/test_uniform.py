import sys
sys.path.append("/home/yourself/fog-computing-orchestration")

from tools import utils

def test_poisson():
	utils.initRandom()

	samples = []
	for i in range(0, 1000):
		samples.append(utils.uniformRandom(3.6))

	assert max(samples) < 3.6
	assert min(samples) > 0
	assert round(utils.listavg(samples), 2) == 1.8