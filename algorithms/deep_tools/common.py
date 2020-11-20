#!/usr/bin/env python


# TODO@luis: re-do this
# General Advantage Estimator
def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
	deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
	deltas = np.stack(deltas)
	gaes = copy.deepcopy(deltas)
	for t in reversed(range(len(deltas) - 1)):
    	gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

	target = gaes + values
	if normalize:
    	gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
	return gaes, target

# build a normal advantage calculation too (?)