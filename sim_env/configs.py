#!/usr/bin/env python
"""This file contains all constants necessary to run the simulations with the specified settings of the Master Thesis presented @ IST"""

# GLOBAL FOG VARIABLES
TIME_STEP = 0.001 # seconds
TOTAL_TIME_STEPS = 1024
SIM_TIME = TOTAL_TIME_STEPS*TIME_STEP
DEBUG = False

# envrionment reward related
OVERLOAD_WEIGHT = 2 # 2 used when no delay constraint

# random related
RANDOM_SEED = 2**19-1 # mersenne prime seeds at 2, 3, 5, 7, 13, 17, 19, 31
RANDOM_SEED_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# NODES CONSTANTS
# transmission
NODE_BANDWIDTH = 1e6 # Hz
NODE_BANDWIDTH_UNIT = 1e5 # Hz , i.e., 10 concurrent transmissions is the maximum
TRANSMISSION_POWER = 20 # dBm
# resources
MAX_QUEUE = 10
CPU_CLOCKS = [5e9, 6e9, 7e9, 8e9, 9e9, 10e9] # [5, 6, 7, 8, 9, 10] GHz
RAM_SIZES = [2400, 4000, 8000] # MB = [6, 10, 20] units
CPU_UNIT = 1e9 # 1 GHz
RAM_UNIT = 400 # MB

# ENV CONSTANTS
AREA = [100, 100] # m x m
PATH_LOSS_EXPONENT = 4
PATH_LOSS_CONSTANT = 0.001
THERMAL_NOISE_DENSITY = -174 # dBm/Hz

# TASK CONSTANTS
PACKET_SIZE = 5000 # bits = 5 kBits

# NOTE: this has to change according to the TEST cases used!!
DEFAULT_SLICES = 1
N_NODES = 5 # default = 5

# experimental case
BASE_SLICE_CHARS = {
	"case": "base",
	"arrivals" :  [0.6, 0.6], #[0.6, 0.6, 0.6], # [0.6],
	"task_type" : [[10, 600, 800], [20, 1200, 400]]# [[5, 600, 400], [10, 600, 400], [10, 400, 800]] # [[15, 1200, 800]] 
}

# --- simulation cases writing ---
# to test offloading
OFF_CASE_1 = {
	"case": "ofc1",
	"arrivals" :  [0.6],
	"task_type" : [[1000, 1200, 800]]
}
H_OFF_CASE_1 = {
	"case": "hofc1",
	"arrivals" :  [0.7],
	"task_type" : [[1000, 1200, 800]]
}
HH_OFF_CASE_1 = {
	"case": "h2ofc1",
	"arrivals" :  [0.7],
	"task_type" : [[1000, 1200, 800]]
}

OFF_CASE_2 = {
	"case": "ofc2",
	"arrivals" :  [0.6],
	"task_type" : [[15, 1200, 800]]
}


# for 2 slices: to test offloading & scheduling + task heterogenity
OFF_SCH_CASE_1 = {
	"case": "offsch1",
	"arrivals" : [0.6, 0.6],
	"task_type" : [[1000, 400, 1200], [1000, 1200, 400]]
}

OFF_SCH_CASE_2 = {
	"case": "offsch2",
	"arrivals" : [0.6, 0.6],
	"task_type" : [[10, 400, 1200], [50, 1200, 400]]
}