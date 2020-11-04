#!/usr/bin/env python

import numpy as np

from utils.tools import uniform_rand_float

def db_to_linear(value):
    return 10**(0.1*value)

def linear_to_db(value):
    if value <= 0: return float("-inf")
    return 10*np.log10(value)

def channel_gain(distance, linear_coefficient, exponential_coefficient):
    if distance <= 0: return 0
    return linear_coefficient*distance**(-exponential_coefficient)
            
def shannon_hartley(gain, power, bandwidth, noise_density):
    if bandwidth < 0 or power < 0 or gain < 0 or noise_density < 0: return 0
    return bandwidth*np.log2(1+((gain*power)/(bandwidth*noise_density))) 

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def bernoulli_arrival(p):
	if not (0<=p<=1): return 0
	return True if uniform_rand_float() < p else False
