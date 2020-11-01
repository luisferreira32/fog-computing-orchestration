# GLOBAL FOG VARIABLES
TIME_STEP = 0.001 # seconds
SIM_TIME_STEPS = 5000*TIME_STEP # 5 seconds actually
N_NODES = 5

# NODES CONSTANTS
# transmission
NODE_BANDWIDTH = 1 # MHz
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
	"task_type" : [[1, 1, 0], [1, 1, 0], [1, 1, 0]]
}

# ENV CONSTANTS
AREA = [100, 100] # m x m
PATH_LOSS_EXPONENT = 4
PATH_LOSS_CONSTANT = 0.001
THERMAL_NOISE_DENSITY = -174 # dBm/Hz

# TASK CONSTANTS
PACKET_SIZE = 5 # Kbit
DEADLINES = [10, 50, 100] # ms
CPU_DEMANDS = [200, 400, 600] # cycles/bit
RAM_DEMANDS = [400, 1200] # MB

