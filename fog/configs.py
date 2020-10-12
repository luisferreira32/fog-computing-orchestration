# GLOBAL FOG VARIABLES

# auxiliary for the programming
FOG_DEBUG = 0
DISPLAY = False
SIM_TIME = 10000
TIME_INTERVAL = 1
N_NODES = 5

# fog core constants
MAX_QUEUE = 10
MAX_W = 20
SERVICE_RATE = 1.8 # default 1.8
MAX_AREA = (100, 100)
UTILITY_REWARD = 0.5
DEFAULT_CPI = 5

# task constants
DEFAULT_DATA = 500*8 # ?500 Mbytes = 500 * 8 M bits
DEFAULT_IL = 200 # * 10**6 instruction lines
TASK_ARRIVAL_RATE = 5.2 # default 5.2

# comunications constants
DEFAULT_POWER = 20 # dBm
DEFAULT_BANDWIDTH = 2 # MHz
B1_PATHLOSS = 0.001
B2_PATHLOSS = 4
DISTANCE_VECTOR = [] # this one is calculated based on spacial placement
N0 = -174 # thermal noise power density 174 dBm/ Hz? Or mW/ Hz?


# derivated default constantsw
DEFAULT_CPS = SERVICE_RATE*DEFAULT_IL*DEFAULT_CPI # * 10**8 if calculated with all default values


