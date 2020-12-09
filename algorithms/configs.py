# For results to be reproduced - a base seed
ALGORITHM_SEED = 2**3-1 # mersenne primes 2, 3, 5, 7, 13, 17, 19, 31

# From env, we have the global default action space as:
DEFAULT_ACTION_SPACE = [6,6,6,11,11,11]

# RL related variables
DEFAULT_TRAJECTORY = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_ITERATIONS = 30

DEFAULT_LEARNING_RATE = 1e-4 # below 1e-4 to avoid suboptimal convergence
DEFAULT_GAMMA = 0.95

DEFAULT_PPO_EPS = 0.2