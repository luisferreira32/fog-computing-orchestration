#!/usr/bin/env python

# do a policy based RL (just one network) with the ppo loss function
# advantages of PPO are found in a discrete actions and multi-process style; offers fast convergence

# since we're implementing ppo with deep neural networks
from deep_tools import frames, losses

# and to make it reproducible
from configs import ALGORITHM_SEED
frames.set_tf_seed(ALGORITHM_SEED)