#external imports
import math
import random as rd

def appendict(lines=None, key=None, value=None):
	"""appends information to a dictionary called lines
	
	Parameters
	----------
	lines=None
		the dictionary with information
	key=None
		the key of the dictionary
	value=None
		the value to append

	Return
	------
	-1 if failed, 0 otherwise
	"""
	if lines == None or key == None or value == None:
		return -1

	if key in lines:
		lines[key].append(value)
	else:
		lines[key] = [value]

	return 0

def listavg(l=None):
	"""finds the average value on a numeric list, returns 0 if failed (or if average is zero)
	"""
	if l is None:
		return 0
	avg = 0
	try:
		avg = sum(l)/len(l)
	except:
		return 0
	return avg

def distOfEventNumber(lbd):
	"""Returns a list with the poission distribution for 0 - 2*lbd
	"""
	dist = []
	for k in range(0,2*lbd+1):
		dist.append(((lbd**k) * math.exp(-lbd))/(math.factorial(k)))
	return dist

def distOfWaitingTime(lbd, interval):
	""" List distribution of waiting time according to a poisson process
	"""
	dist = []
	for s in range(1, interval+1):
		for ds in range(0,10):
			t = s + 0.1*ds
			dist.append(1-math.exp(-lbd/interval*t))
	return dist

def poissonNextEvent(dist):
	""" gives the next time an event will ocurr acording to the lbd
	"""
	uniformrand = rd.random()
	for i in range(1, len(dist)):
		if uniformrand < dist[i-1]: return i/10
	return i/10

def uniformRandom(m=1):
	""" gives a random number between [0, m]
	"""
	return m*rd.random()

def initRandom():
	""" inits the seed with a defined value
	"""
	rd.seed(10017)