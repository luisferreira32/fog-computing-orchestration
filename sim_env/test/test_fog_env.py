#!/usr/bin/env python

import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
    sys.path.append("/home/yourself/fog-computing-orchestration")
import numpy as np

from sim_env.events import Event_queue, Task_arrival, Offload, Start_processing, Task_finished, Set_arrivals
from sim_env.core_classes import create_random_node, Task
from sim_env.configs import N_NODES, MAX_QUEUE, DEFAULT_SLICES
from sim_env.envrionment import Fog_env, split_observation_by_logical_groups, get_nodes_characteristics

def test_next_observation_basics():
	# create the env that has the nodes
	fe = Fog_env()
	# zeros default observation, and nodes characteristics
	z = np.zeros(DEFAULT_SLICES*N_NODES, dtype=np.uint8)
	(r_c_default, r_m_default) = get_nodes_characteristics(fe.nodes)
	# check the empty initial state
	obs = fe._next_observation()
	[a_ik, b_ik, be_ik, r_ic, r_im] = split_observation_by_logical_groups(obs)
	assert np.all(a_ik == z)
	assert np.all(b_ik == z)
	assert np.all(be_ik == z)
	assert np.all(r_ic == r_c_default)
	assert np.all(r_im == r_m_default)
	# manually insert tasks on this timestep
	fe.nodes[0].add_task_on_slice(0, Task(0))
	fe.nodes[N_NODES-1].add_task_on_slice(0, Task(0))
	# check the new altered state state
	obs = fe._next_observation()
	[a_ik, b_ik, be_ik, r_ic, r_im] = split_observation_by_logical_groups(obs)
	expected_a = [1 if i == 0 or i == (N_NODES-1)*DEFAULT_SLICES else 0 for i in range(DEFAULT_SLICES*N_NODES)]
	expected_b = expected_a
	assert np.all(a_ik == expected_a)
	assert np.all(b_ik == expected_b)
	assert np.all(be_ik == z)
	# insert some more in the next timestep
	fe.clock += 1
	fe.nodes[0].add_task_on_slice(0, Task(1))
	fe.nodes[N_NODES-1].add_task_on_slice(0, Task(1))
	# and check for consistency
	obs = fe._next_observation()
	[a_ik, b_ik, be_ik, r_ic, r_im] = split_observation_by_logical_groups(obs)
	expected_b = [b+1 if b==1 else 0 for b in expected_b]
	assert np.all(a_ik == expected_a)
	assert np.all(b_ik == expected_b)
	assert np.all(be_ik == z)
	# manually start some processing and move the timestep
	fe.clock += 1
	fe.nodes[0].start_processing_in_slice(0, 1)
	# check if only one node is processing and no new arrivals
	obs = fe._next_observation()
	[a_ik, b_ik, be_ik, r_ic, r_im] = split_observation_by_logical_groups(obs)
	expected_be = [1 if i == 0 else 0 for i in range(DEFAULT_SLICES*N_NODES)]
	assert np.all(a_ik == z)
	assert np.all(b_ik == expected_b)
	assert np.all(be_ik == expected_be)