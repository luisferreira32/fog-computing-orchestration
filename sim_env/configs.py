# GLOBAL FOG VARIABLES
TIME_STEP = 0.001 # seconds
TRAINING_STEPS = 3000 # stable results in baselines after 3000
SIM_TIME_STEPS = TRAINING_STEPS*TIME_STEP
RANDOM_SEED = 2**31-1 # mersenne prime seeds at 2, 3, 5, 7, 13, 17, 19, 31
DEBUG = False

# NODES CONSTANTS
N_NODES = 5
# transmission
NODE_BANDWIDTH = 1e6 # Hz
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
	"case": "n1",
	"arrivals" : [0.6, 0.6, 0.6],
	"task_type" : [[10, 400, 400], [10, 600, 400], [10, 400, 1200]]
}

# ENV CONSTANTS
AREA = [100, 100] # m x m
PATH_LOSS_EXPONENT = 4
PATH_LOSS_CONSTANT = 0.001
THERMAL_NOISE_DENSITY = -174 # dBm/Hz

# TASK CONSTANTS
PACKET_SIZE = 5000 # bit
# this values are directly described in the simulation cases
# DEADLINES = [10, 50, 100] # ms
# CPU_DEMANDS = [200, 400, 600] # cycles/bit
# RAM_DEMANDS = [400, 1200] # MB

# ---- simulation cases ---- 

# Case 1
NORMAL_CASE_1 = BASE_SLICE_CHARS
HEAVY_CASE_1 ={
	"case": "h1",
	"arrivals" : [0.8, 0.8, 0.8],
	"task_type" : [[10, 400, 400], [10, 600, 400], [10, 400, 1200]]
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