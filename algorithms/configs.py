import os

# For results to be reproduced - a base seed
ALGORITHM_SEED = 2**3-1 # mersenne primes 2, 3, 5, 7, 13, 17, 19, 31

# RL related variables
DEFAULT_BATCH_SIZE = 32
DEFAULT_GAMMA = 0.98
RW_EPS = 0.02

# ppo configs
DEFAULT_TRAJECTORY = 17 # +1 since TD error; 33, 65, 129
DEFAULT_EPOCHS = 3
DEFAULT_ITERATIONS = 700
DEFAULT_PPO_EPS = 0.2
PARALLEL_ENVS = 4
ENV_RESET_IT = 40# default 10
DEFAULT_PPO_LEARNING_RATE = 1e-2
DEFAULT_CRITIC_LEARNING_RATE = 1e-4

# dqn configs
DEFAULT_DQN_LEARNING_RATE = 1e-3
MIN_DQN_LEARNING_RATE = 1e-3
MAX_DQN_TRAIN_ITERATIONS = 10000 # default 2e4
REPLAY_BUFFER_SIZE = 5000 # default 1e4
TARGET_NETWORK_UPDATE_RATE = 2000 # default 1e3

EPSILON_RENEWAL_RATE = 5000 # default 5e3
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_RENEWAL_FACTOR = 0.9



# time frame configs
TIME_SEQUENCE_SIZE = 10

# for saving
DEFAULT_SAVE_MODELS_PATH = os.getcwd()+"/algorithms/saved_models/"