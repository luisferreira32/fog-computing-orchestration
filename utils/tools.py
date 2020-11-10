#!/usr/bin/env python

#external imports
from numpy import random
import numpy as np
import pandas as pd
import os

# --- random related tools ---

def set_seed(seed=None):
	np.random.seed(seed)

def uniform_rand_float(m=1):
	return np.random.rand()*m

def uniform_rand_int(low=0, high=1):
	return np.random.randint(low=low, high=high)

def uniform_rand_choice(l):
	return np.random.choice(l)

def random_seed_primes(max_val=100):
	return [num for num in range(2, max_val) if all(num%i!=0 for i in range(2,int(np.sqrt(num))+1)) ]

# --- data structure related tools ---

def dictionary_append(d, key, info):
	if key in d:
		d[key] = np.append(d[key],info)
	else:
		d[key] = info
	return d

# --- I/O related tools ---

results_path = os.getcwd()+"/results/"

def format_delay(delay_dict):
	df = pd.DataFrame.from_dict(delay_dict)
	return df.describe()

def no_format(data):
	return data

def append_to_file(filename, data, format_data=no_format):
	with open(results_path+filename, "a") as f:
		d = format_data(data)
		f.write(str(d))
	return
