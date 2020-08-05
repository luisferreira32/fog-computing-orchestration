# GLOBAL FOG VARIABLES

# auxiliary for the programming
FOG_DEBUG = 1

# fog core constants
MAX_QUEUE = 10
MAX_INFLUX = 10
SERVICE_RATE = 1.8
MAX_AREA = (100, 100)
UTILITY_REWARD = 0.5
DEFAULT_CPI = 5

# task constants
DEFAULT_DATA = 500 # Mbytes
DEFAULT_IL = 200 # * 10**8 instruction lines
TASK_ARRIVAL_RATE = 5.2

# comunications constants
DEFAULT_POWER = 20 # dBm
DEFAULT_BANDWIDTH = 2 # MHz
B1_PATHLOSS = 0.001
B2_PATHLOSS = 4
DISTANCE_VECTOR = [] # this one is calculated based on spacial placement
N0 = 174 # thermal noise power 174 dBm/Hz


# derivated default constantsw
DEFAULT_CPS = SERVICE_RATE*DEFAULT_IL*DEFAULT_CPI # * 10**8 if calculated with all default values
