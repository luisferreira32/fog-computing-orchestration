# GLOBAL FOG VARIABLES
TIME_STEP = 0.001 # seconds
SIM_TIME_STEPS = 5000*TIME_STEP # 5 seconds actually
TRAINING_STEPS = 5000
RANDOM_SEED = 10017

# NODES CONSTANTS
N_NODES = 5
# transmission
NODE_BANDWIDTH = 1*10**6 # Hz
TRANSMISSION_POWER = 20 # dBm
# resources
DEFAULT_SLICES = 3
MAX_QUEUE = 10
CPU_CLOCKS = [5, 6, 7, 8, 9, 10] # GHz
RAM_SIZES = [2400, 4000, 8000] # MB
CPU_UNIT = 1 # GHz
RAM_UNIT = 400 # MB

# slice characteristics 
BASE_SLICE_CHARS = {
	"arrivals" : [0.5, 0.5, 0.5],
	"task_type" : [[0, 1, 0], [0, 2, 0], [0, 1, 1]]
}

# ENV CONSTANTS
AREA = [100, 100] # m x m
PATH_LOSS_EXPONENT = 4
PATH_LOSS_CONSTANT = 0.001
THERMAL_NOISE_DENSITY = -174 # dBm/Hz

# TASK CONSTANTS
PACKET_SIZE = 5000 # bit
DEADLINES = [10, 50, 100] # ms
CPU_DEMANDS = [200, 400, 600] # cycles/bit
RAM_DEMANDS = [400, 1200] # MB

# ---- simulation cases ---- 

# Case A
NORMAL_CASE_A = BASE_SLICE_CHARS
HEAVY_CASE_A ={
	"arrivals" : [0.9, 0.9, 0.9],
	"task_type" : [[0, 1, 0], [0, 2, 0], [0, 1, 1]]
}
# Case B
NORMAL_CASE_B = {
	"arrivals" : [0.5, 0.5, 0.5],
	"task_type" : [[0, 1, 0], [1, 1, 0], [2, 1, 0]]
}
HEAVY_CASE_B = {
	"arrivals" : [0.9, 0.9, 0.9],
	"task_type" : [[0, 1, 0], [1, 1, 0], [2, 1, 0]]
}
# Case C
NORMAL_CASE_C = {
	"arrivals" : [0.5, 0.5, 0.5],
	"task_type" : [[0, 1, 0], [1, 2, 0], [1, 1, 0]]
}
HEAVY_CASE_C = {
	"arrivals" : [0.9, 0.9, 0.9],
	"task_type" : [[0, 1, 0], [1, 2, 0], [1, 1, 0]]
}