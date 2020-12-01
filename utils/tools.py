#!/usr/bin/env python

#external imports
import numpy as np
import pandas as pd
import os
import csv

# --- random related tools ---

def set_tools_seed(seed=None):
	global np_tools_random # change the np_tools_random seed
	np_tools_random = np.random.RandomState(seed=seed)

def uniform_rand_float(m=1):
	return np_tools_random.rand()*m

def uniform_rand_int(low=0, high=1):
	return np_tools_random.randint(low=low, high=high)

def uniform_rand_array(size=1):
	return np_tools_random.random(size)

def uniform_rand_choice(l):
	return np_tools_random.choice(l)

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
windows_path = os.getcwd()+"\\results\\"

# format types
def format_to_miliseconds(data):
	return np.array([d*1000 for d in data])

def no_format(data):
	return data

# aux functions
def write_dictionary_on_csvs(d, format_data=no_format):
	for key, value in d.items():
		write_to_csv(key+".csv", value, format_data)

# file operations - note newline='' is for the windows csv avoid double \n
def append_to_file(filename, data, format_data=no_format):
	with open(windows_path+filename, "a",newline='') as f:
		d = format_data(data)
		f.write(str(d))
	return

def write_to_csv(filename, data, format_data=no_format):
	with open(windows_path+filename, "w",newline='') as f:
		d = format_data(data)
		wr = csv.writer(f)
		wr.writerow(d)
	return

# hardcoded for this work
def write_all_to_csvs(delay_df, success_df, overflow_df):
	for key in delay_df:
		d_n= format_to_miliseconds(delay_df[key])
		s_n = success_df[key]
		o_n = overflow_df[key]
		with open(windows_path+key+".csv","w",newline='') as f:
			wr = csv.writer(f)
			wr.writerows([["average_delay_ms",d] for d in d_n])
			wr.writerows([["success_rate",s] for s in s_n])
			wr.writerows([["overflow_rate",o] for o in o_n])

# hard coded for this work too
def read_all_from_csvs(path_to_files=windows_path, cases=["n1","n2","n3","h1","h2","h3"]):
	# create necessary dicts
	delay_df = {}
	success_df = {}
	overflow_df = {}

	# check all things in the results directory
	for name in os.listdir(path_to_files):
		# check if it's one of the cases
		if not any([case in name for case in cases]):
			continue
		if name.endswith(".csv"):
			key = name[:-4] # without the csv
			with open(path_to_files+name) as f:
				reader = csv.reader(f)
				for row in reader:
					if row[0] == "average_delay_ms":
						delay_df = dictionary_append(delay_df, key, float(row[1]))
					elif row[0] == "success_rate":
						success_df = dictionary_append(success_df, key, float(row[1]))
					elif row[0] == "overflow_rate":
						overflow_df = dictionary_append(overflow_df, key, float(row[1]))
	return delay_df, success_df, overflow_df