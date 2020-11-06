#!/usr/bin/env python

#external imports
from numpy import random
import numpy as np

def set_seed(seed=None):
	np.random.seed(seed)

def uniform_rand_float(m=1):
	return np.random.rand()*m

def uniform_rand_int(low=0, high=1):
	return np.random.randint(low=low, high=high)

def uniform_rand_choice(l):
	return np.random.choice(l)
