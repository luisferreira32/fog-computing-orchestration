#external imports
import math

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

def poissondist(lbd=5):
	"""Returns a list with the poission distribution for 0 - 2*lbd
	"""
	dist = []
	for k in range(0,2*lbd+1):
		dist.append(((lbd**k) * math.exp(-lbd))/(math.factorial(k)))
	return dist

def discreteX(distribution, normalrand):
	"""Returns the event X in which the probability landed
	"""
	psum = 0
	for x in range(0,len(distribution)):
		psum += distribution[x]
		if normalrand < psum:
			return x
	return len(distribution)