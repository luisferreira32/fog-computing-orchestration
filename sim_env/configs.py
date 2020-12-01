#!/usr/bin/env python
"""This file contains all constants necessary to run the simulations with the specified settings of the Master Thesis presented @ IST"""

# GLOBAL FOG VARIABLES
TIME_STEP = 0.001 # seconds
TOTAL_TIME_STEPS = 1000
SIM_TIME = TOTAL_TIME_STEPS*TIME_STEP
DEBUG = False

# envrionment reward related
OVERLOAD_WEIGHT = 2

# random related
RANDOM_SEED = 2**19-1 # mersenne prime seeds at 2, 3, 5, 7, 13, 17, 19, 31
RANDOM_SEED_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# NODES CONSTANTS
N_NODES = 5
# transmission
NODE_BANDWIDTH = 1e6 # Hz
TRANSMISSION_POWER = 20 # dBm
# resources
DEFAULT_SLICES = 3
MAX_QUEUE = 10
CPU_CLOCKS = [5e9, 6e9, 7e9, 8e9, 9e9, 10e9] # [5, 6, 7, 8, 9, 10] GHz
RAM_SIZES = [2400, 4000, 8000] # MB = [6, 10, 20] units
CPU_UNIT = 1e9 # 1 GHz
RAM_UNIT = 400 # MB

# slice characteristics 
BASE_SLICE_CHARS = {
	"case": "n1",
	"arrivals" : [0.6, 0.6, 0.6],
	"task_type" : [[10, 400, 400], [10, 600, 400], [10, 200, 1200]]
}

# ENV CONSTANTS
AREA = [100, 100] # m x m
PATH_LOSS_EXPONENT = 4
PATH_LOSS_CONSTANT = 0.001
THERMAL_NOISE_DENSITY = -174 # dBm/Hz

# TASK CONSTANTS
PACKET_SIZE = 5000 # bits = 5 kBits (7, 10?)

# ---- simulation cases ---- 
# Case 1
NORMAL_CASE_1 = BASE_SLICE_CHARS
HEAVY_CASE_1 ={
	"case": "h1",
	"arrivals" : [0.8, 0.8, 0.8],
	"task_type" : [[10, 400, 400], [10, 600, 400], [10, 200, 1200]]
}
# Case 2
NORMAL_CASE_2 = {
	"case": "n2",
	"arrivals" : [0.6, 0.6, 0.6],
	"task_type" : [[10, 400, 400], [50, 400, 400], [100, 400, 400]]
}
HEAVY_CASE_2 = {
	"case": "h2",
	"arrivals" : [0.8, 0.8, 0.8],
	"task_type" : [[10, 400, 400], [50, 400, 400], [100, 400, 400]]
}
# Case 3
NORMAL_CASE_3 = {
	"case": "n3",
	"arrivals" : [0.6, 0.6, 0.6],
	"task_type" : [[10, 400, 400], [50, 600, 400], [50, 400, 400]]
}
HEAVY_CASE_3 = {
	"case": "h3",
	"arrivals" : [0.8, 0.8, 0.8],
	"task_type" : [[10, 400, 400], [50, 600, 400], [50, 400, 400]]
}
# cases compilations
ALL_CASES = [NORMAL_CASE_1, NORMAL_CASE_2, NORMAL_CASE_3, HEAVY_CASE_1, HEAVY_CASE_2, HEAVY_CASE_3]
HEAVY_CASES = [HEAVY_CASE_1, HEAVY_CASE_2, HEAVY_CASE_3]
NORMAL_CASES = [NORMAL_CASE_1, NORMAL_CASE_2, NORMAL_CASE_3]
