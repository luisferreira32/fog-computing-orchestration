# GLOBAL FOG VARIABLES
TIME_STEP = 0.001 # seconds
TRAINING_STEPS = 5000
SIM_TIME_STEPS = TRAINING_STEPS*TIME_STEP
RANDOM_SEED = 2**13-1 # mersenne prime seeds at 2, 3, 5, 7, 13, 17, 19, 31
DEBUG = False

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
	"case": "n_case_1",
	"arrivals" : [0.6, 0.6, 0.6],
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

# Case 1
NORMAL_CASE_1 = BASE_SLICE_CHARS
HEAVY_CASE_1 ={
	"case": "h_case_1",
	"arrivals" : [0.8, 0.8, 0.8],
	"task_type" : [[0, 1, 0], [0, 2, 0], [0, 1, 1]]
}
# Case 2
NORMAL_CASE_2 = {
	"case": "n_case_2",
	"arrivals" : [0.6, 0.6, 0.6],
	"task_type" : [[0, 1, 0], [1, 1, 0], [2, 1, 0]]
}
HEAVY_CASE_2 = {
	"case": "h_case_2",
	"arrivals" : [0.8, 0.8, 0.8],
	"task_type" : [[0, 1, 0], [1, 1, 0], [2, 1, 0]]
}
# Case 3
NORMAL_CASE_3 = {
	"case": "n_case_3",
	"arrivals" : [0.6, 0.6, 0.6],
	"task_type" : [[0, 1, 0], [1, 2, 0], [1, 1, 0]]
}
HEAVY_CASE_3 = {
	"case": "h_case_3",
	"arrivals" : [0.8, 0.8, 0.8],
	"task_type" : [[0, 1, 0], [1, 2, 0], [1, 1, 0]]
}