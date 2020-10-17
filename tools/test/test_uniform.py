import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
	sys.path.append("/home/yourself/fog-computing-orchestration")

from tools import utils

def test_uniform():
	utils.initRandom()

	samples = []
	for i in range(0, 1000):
		samples.append(utils.uniformRandom(3.6))

	assert max(samples) < 3.6
	assert min(samples) > 0
	assert round(utils.listavg(samples), 2) == 1.8

def test_seed_on_uniform():
	utils.initRandom()
	a = utils.uniformRandom()
	utils.initRandom()
	assert a == utils.uniformRandom()

	y = [0, 1]
	utils.initRandom()
	a = utils.randomChoice(y)
	utils.initRandom()
	assert a == utils.randomChoice(y)

def test_random_uniform_choice():
	utils.initRandom()
	total = 10000
	a = [1,2]; ct0 = 0; ct1 = 0
	for i in range(total):
		x = utils.randomChoice(a)
		if x == a[0]: ct0 +=1
		if x == a[1]: ct1 +=1
	assert round(ct0/total,1) == round(ct1/total,1)