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

def append_to_file(filename, data, format_data=format_delay):
	with open(results_path+filename, "a") as f:
		d = format_data(data)
		f.write(str(d))
	return
