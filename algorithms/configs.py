import os

# For results to be reproduced - a base seed
ALGORITHM_SEED = 2**3-1 # mersenne primes 2, 3, 5, 7, 13, 17, 19, 31

# RL related variables
DEFAULT_BATCH_SIZE = 32
DEFAULT_GAMMA = 0.98

# ppo configs
DEFAULT_TRAJECTORY = 65 # +1 since TD error
DEFAULT_EPOCHS = 3 #10?
DEFAULT_ITERATIONS = 300 # 100?
DEFAULT_PPO_EPS = 0.2
PARALLEL_ENVS = 3
DEFAULT_PPO_LEARNING_RATE = 1e-2
DEFAULT_CRITIC_LEARNING_RATE = 1e-4

# dqn configs
DEFAULT_DQN_LEARNING_RATE = 1e-3
MAX_DQN_TRAIN_ITERATIONS = 100 #1e4
TARGET_NETWORK_UPDATE_RATE = 100 # so updates 10 times
REPLAY_BUFFER_SIZE = 1000 #1e4

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_RENEWAL_RATE = 1000 # 5000
EPSILON_RENEWAL_FACTOR = 0.9



# time frame configs
TIME_SEQUENCE_SIZE = 10

# for saving
DEFAULT_SAVE_MODELS_PATH = os.getcwd()+"/algorithms/saved_models/"