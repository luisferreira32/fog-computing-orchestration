#!/usr/bin/env python

#external imports
from numpy import random
import numpy as np
import pandas as pd
import os
import csv

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

# format types
def format_to_miliseconds(data):
	return np.array([d*1000 for d in data])

def no_format(data):
	return data

# aux functions
def write_dictionary_on_csvs(d, format_data=no_format):
	for key, value in d.items():
		write_to_csv(key+".csv", value, format_data)

# file operations
def append_to_file(filename, data, format_data=no_format):
	with open(results_path+filename, "a") as f:
		d = format_data(data)
		f.write(str(d))
	return

def write_to_csv(filename, data, format_data=no_format):
	with open(results_path+filename, "w") as f:
		d = format_data(data)
		wr = csv.writer(f)
		wr.writerow(d)
	return

# HARDCODE - just for this work
def write_all_to_csvs(delay_df, success_df, overflow_df):
	for key in delay_df:
		d_n= format_to_miliseconds(delay_df[key])
		s_n = success_df[key]
		o_n = overflow_df[key]
		with open(results_path+key+".csv","w") as f:
			wr = csv.writer(f)
			wr.writerows([["average_delay_ms",d] for d in d_n])
			wr.writerows([["success_rate",s] for s in s_n])
			wr.writerows([["overflow_rate",o] for o in o_n])
