import os

# For results to be reproduced - a base seed
ALGORITHM_SEED = 2**3-1 # mersenne primes 2, 3, 5, 7, 13, 17, 19, 31

# RL related variables
DEFAULT_TRAJECTORY = 128
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 1
DEFAULT_ITERATIONS = 100

DEFAULT_PPO_LEARNING_RATE = 1e-2
DEFAULT_CRITIC_LEARNING_RATE = 1e-3
DEFAULT_GAMMA = 0.95

DEFAULT_PPO_EPS = 0.2
PARALEL_ENVS = 3

TIME_SEQUENCE_SIZE = 10

# for saving
DEFAULT_SAVE_MODELS_PATH = os.getcwd()+"/algorithms/saved_models/"