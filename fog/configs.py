# GLOBAL FOG VARIABLES

# auxiliary for the programming
FOG_DEBUG = 1

# fog core constants
MAX_QUEUE = 10
MAX_INFLUX = 10
SERVICE_RATE = 5
MAX_AREA = (100, 100)

# task constants
DEFAULT_DATA = 500 # Mbytes
DEFAULT_IL = 200 # * 10^ factor
DEFAULT_FACTOR = 8

# comunications constants
DEFAULT_POWER = 20 # dBm
DEFAULT_BANDWIDTH = 2 # MHz
TASK_ARRIVAL_RATE = 3
B1_PATHLOSS = 0.001
B2_PATHLOSS = 4
DISTANCE_VECTOR = [] # this one is calculated based on spacial placement
N0 = 174 # thermal noise power 174 dBm/Hz
