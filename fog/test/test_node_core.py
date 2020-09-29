import sys
sys.path.append("/home/yourself/fog-computing-orchestration")

from fog import configs
from fog import node
from fog import coms
from tools import utils

def test_node_creation():
	n1 = node.Core()
	# check all default values and some basic functions
	assert n1.cpi == configs.DEFAULT_CPI
	assert n1.cps == configs.DEFAULT_CPS
	assert n1.processing == False
	assert n1.bw == configs.DEFAULT_BANDWIDTH
	assert n1.pw == configs.DEFAULT_POWER
	assert n1.qs() == 0
	assert n1.emptyqueue() == True
	assert n1.fullqueue() == False
	# if parameters change, atributes change
	n2 = node.Core(cpu=(1,2))
	assert n2.cpi == 1
	assert n2.cps == 2

def test_cpu_queue():
	n1 = node.Core()
	t1 = coms.Task(0)

	while not n1.fullqueue():
		# if there is still space in the queue, it should not discard
		assert n1.queue(t1) == None
	assert n1.queue(t1) == t1

	while not n1.emptyqueue():
		# while t1 is in the queue, everytime we process, should return t1
		assert n1.process(1) == t1
		assert t1.completed == True
		assert t1.delay == 1 + 1/configs.SERVICE_RATE # starting process + process time = 1+1/SR
	assert n1.process(1) == None

def test_edges_creation():
	utils.initRandom()
	nodes = []
	for i in range(0, configs.N_NODES):
		n = node.Core(name="n"+str(i), 
			placement=(utils.uniformRandom(configs.MAX_AREA[0]),utils.uniformRandom(configs.MAX_AREA[1])),
			cpu=(configs.DEFAULT_CPI, configs.DEFAULT_CPS))
		nodes.append(n)
	# create M edges between each two nodes
	for n in nodes:
		n.setcomtime(nodes)

	for n1 in nodes:
		for n2 in nodes:
			if n1 == n2: continue
			# check if there are edges and the time is between 0 and infinite
			assert n1.comtime[n2] != None
			assert n1.comtime[n2] > 0
			assert n1.comtime[n2] < 99999