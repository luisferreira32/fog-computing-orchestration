#!/usr/bin/env python
"""Simple calculators used for a fog envrionment necessities, provides functions for:

plane assumptions there is an euclidean distance calculator;
unit conversion from dB and to dB there are two functions;
a channel gain calculated with a linear an exponential coefficients;
shannon-hartley bit rate theorem given a gain, transmission power, bandwidth and noise density.
"""

# >>>>> imports
import numpy as np
from utils.tools import uniform_rand_float # to use a seeded random in the simulations

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> classes and functions

eps = np.finfo(np.float32).eps.item() # almost zero value

def db_to_linear(value):
	"""Returns a linear value given its dB value """
	return 10**(0.1*float(value))

def linear_to_db(value):
	"""Returns a dB value given its linear value """
	if value <= 0:
		return None
	return 10*np.log10(float(value))

def channel_gain(distance, linear_coefficient, exponential_coefficient):
	"""Returns a medium distance (>1m) channel gain given the distance between two nodes and the coefficients

	A linear and an exponential coefficients all greater than zero, return a channel gain in [0,1]
		channel gain = linear coefficient * distance ^ (- exponential coefficient)

	Parameters
	----------
	distance: float
		channel transmission distance
	linear_coefficient: float
		linear coefficient greater than zero
	exponential_coefficient: float
		exponential coefficient greater than zero

	"""
	if distance <= 0 or linear_coefficient <= 0 or exponential_coefficient <= 0:
		return 0
	return linear_coefficient*distance**(-exponential_coefficient)
            
def shannon_hartley(gain, power, bandwidth, noise_density):
	"""Returns a channel bitrate calculated by the Shannon-Hartley theorem
	
	bitrate = bandwidth * log2(1 + gain * power / bandwidth * noise_density)

	Parameters
	----------
	gain: float
		channel gain, value between [0,1]
	power: float
		transmission power in linear units (mW)
	bandwidth: float
		transmission bandwidth of this noise channel (Hz)
	noise_density: float
		noise density in linear units (mW/Hz)
	"""
	if bandwidth <= 0 or power <= 0 or gain <= 0 or gain > 1 or noise_density <= 0:
		return 0
	return float(bandwidth)*np.log2(1+((float(gain)*float(power))/(float(bandwidth)*float(noise_density)+eps))) 

def euclidean_distance(x1, y1, x2, y2):
	"""Returns the euclidean distance between two sets of points in space: x1,y1,x2,y2 """
	return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def bernoulli_arrival(p):
	"""Calculates if a task arrived according to the Bernoulli distribution with probability p """
	if not (0<=p<=1):
		return 0
	return True if uniform_rand_float() < p else False

# <<<<<
