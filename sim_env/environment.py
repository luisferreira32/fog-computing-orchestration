#!/usr/bin/env python
""" In this file a subclass of a openAI gym envrionment is created to simulate a Fog envrionemnt.

The gym envrionment works in timesteps, recieving an action within the defined action space and returning
an observation of the envrionment in which the actor will base its decision on. The Fog envrionment itself
is modelled by the classes described in sim_env.fog, each step runs events within the timestep, based on
the events described in sim_env.events. This way this Fog envrionment runs as a Discrete Event Simulator.
"""

# >>>>> imports
# external openAI imports
import gym
from gym import spaces
from gym.utils import seeding

# and necessary math libs
import numpy as np

# fog related imports
from sim_env.fog import create_random_node, point_to_point_transmission_rate
from sim_env.events import Event_queue, is_arrival_on_slice
from sim_env.events import Stop_transmitting, Set_arrivals, Offload_task, Start_transmitting, Start_processing
from sim_env.configs import TIME_STEP, SIM_TIME, RANDOM_SEED, OVERLOAD_WEIGHT
from sim_env.configs import N_NODES, DEFAULT_SLICES, MAX_QUEUE, CPU_UNIT, RAM_UNIT
from sim_env.configs import PACKET_SIZE, BASE_SLICE_CHARS, NODE_BANDWIDTH_UNIT

# for reproductibility
from utils.tools import set_tools_seed

# <<<<<
# >>>>> meta-data
__author__ = "Luis Ferreira @ IST"

# <<<<<
# >>>>> envrionment class
class Fog_env(gym.Env):
	""" Fog_env looks to replicate a FC envrionment, configured in sim_env.configs it is a custom environment that follows gym interface

	Attributes:
		(super) np_random: RandomState - used for reproductibility within the envrionment random executions
		nodes: List[Fog_node] - a list of fog nodes that physically exist in the envrionment
		case: dict - the case that defines slices characteristics and arrival rates
		evq: Event_queue - the event queue of the Discrete Event simulator
		
		clock: float - keep track of the simulated time
		max_time: float - the maximum clock value
		time_step: float - clock increment one ach step

		saved_step_info: dict - for render purposes saves last step
		action_space: spaces.MultiDiscrete - the action space limits (N agents/nodes x M actions)
		observation_space: spaces.MultiDiscrete - the observation space limit (N agents/nodes)
	"""

	metadata = {'render.modes': ['human']}

	def __init__(self, case=BASE_SLICE_CHARS, rd_seed=RANDOM_SEED, max_time=SIM_TIME, time_step=TIME_STEP):
		"""
		Parameters:
			case: dict - the case that defines slices characteristics and arrival rates
			rd_seed: int - the seed that this envrionment will use for all pseudo-randoms
		"""

		super(Fog_env, self).__init__()
		# Set up seeds for reproductibility
		self.rd_seed = rd_seed
		self.seed(rd_seed)

		# envrionment variables
		# self.nodes, self.evq, etc...
		self.nodes = [create_random_node(i) for i in range(1,N_NODES+1)]
		self.case = case
		for n in self.nodes:
			n.set_distances(self.nodes)
		self.evq = Event_queue()

		self.clock = 0
		self.max_time = max_time
		self.time_step = time_step

		self.saved_step_info = None

		# define the action space with I nodes and K slices each
		# [[f_00, ..., f_0k, w_00, ..., w_0k], ..., [f_i0, ..., f_ik, w_i0, ..., w_ik]]
		# for each node there is an action [f_i0, ..., f_ik, w_i0, ..., w_ik]
		# where values can be between 0 and I for f_ik, and 0 and N=limited by either memory or cpu for w_ik
		action_possibilities = [np.append([N_NODES+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units, n._avail_ram_units/np.ceil(case["task_type"][2][k]/RAM_UNIT))+1 for k in range(n.max_k)]) for n in self.nodes]
		action_possibilities = np.array(action_possibilities, dtype=np.uint8)
		self.action_space = spaces.MultiDiscrete(action_possibilities)

		# and the state space with I nodes and K slices each
		# [a_00, ..., a_0k, b_00, ..., b_0k, be_00, ..., be_0k, rc_0, rm_0,
		# ..., a_i0, ..., a_ik, b_i0, ..., b_ik, be_i0, ..., be_ik, rc_i, rm_i]
		# state_lows has to be remade if nodes don't have same slices
		state_possibilities = [np.concatenate(([2 for _ in range(n.max_k)],[MAX_QUEUE+1 for _ in range(n.max_k)],
			[min(n._avail_cpu_units,  n._avail_ram_units/np.ceil(case["task_type"][2][k]/RAM_UNIT))+1 for k in range(n.max_k)],
			[n._avail_cpu_units+1], [n._avail_ram_units+1])) for n in self.nodes]
		state_possibilities = np.array(state_possibilities, dtype=np.uint8)
		self.observation_space = spaces.MultiDiscrete(state_possibilities)

		# and the first event that will trigger subsequent arrivals
		self.evq.add_event(Set_arrivals(0, time_step, self.nodes, self.case))

	def step(self, action_n):
		""" Method that provices a step in time, given an action_n, decided by the N agents/nodes, and returns an observaiton.

		Parameters:
			action_n: np.array in action_space - the action decided by each of the nodes to give in this timestep
		Returns:
			obs_n: np.array in observation_space - the observation after the time step ocurred
			reward_n: float - the total reward of the timestep (sum of all agents rewards for their actions)
			done: bool - indicates weather or not this envrionment is out of events to simulate, i.e. ended the simulation time
			info_n: dict - a compiling of information for display
		"""

		# to make sure you give actions in the FORMATED action space
		action_n = self._cap_action_n(action_n)
		assert self.action_space.contains(action_n)
		state_t = self._get_state_obs()

		# to return it's necessary to return a lists in
		obs_n = [] # observations (POMDP)
		reward_n = [] # local rewards
		info_n = {} # information per agent

		# update some envrionment values
		for n in self.nodes:
			n.new_interval_update_service_rate()

		# measure instant reward of an action taken and queue it
		for i in range(N_NODES):
			# set the zeroed info
			info_n[i] = {
				"delay_list" : [],
				"overflow" : 0,
				"discarded" : 0,
				};

			# calculate the instant rewards, based on state, action pair
			rw = self._agent_reward_fun(self.nodes[i], self._get_agent_observation(n), action_n[i])
			reward_n.append(rw)

			# and execute the action
			self._set_agent_action(self.nodes[i], action_n[i]) 

		# increase the clock a timestep
		self.clock += self.time_step
		done = self.clock >= self.max_time
		# rollout the events until new timestep
		while self.evq.has_events() and self.evq.first_time() <= self.clock:
			ev = self.evq.pop_event()
			t = ev.execute(self.evq)
			#print(ev.classtype+"["+str(round(ev.time*1000,2))+"ms]", end='-->')

			# --- GET INFORMAITON HERE ---
			if t is not None: # means it came from a node
				if t.is_completed(): # finished
					info_n[n.index-1]["delay_list"].append(t.task_delay())
				elif t.task_time() == ev.time: # overflowed
					info_n[n.index-1]["overflow"] += 1
				else: # discarded because of delay constraint
					info_n[n.index-1]["discarded"] += 1


		# obtain next observation
		obs_n = self._get_state_obs()

		# just save it for render
		self.saved_step_info = [state_t, action_n]

		# joint optimization -> reward sum
		#assert self.observation_space.contains(obs_n)
		return obs_n, sum(reward_n), done, info_n


	def reset(self):
		""" Resets the state of the envrionment, reseting the Event_queue object and adding the first event trigger, the nodes and the clock """

		# evq reset empties the queue and sets internal clock to zero
		self.evq.reset()
		self.evq.add_event(Set_arrivals(0, self.time_step, self.nodes, self.case))
		for node in self.nodes:
			node.reset() # empty buffers
		self.clock = 0 # and zero simulation clock
		return self._get_state_obs()

	def _get_state_obs(self):
		""" Gets the whole state observation by obtaining POMDP for each agent """

		return np.array([self._get_agent_observation(n) for n in self.nodes], dtype=np.uint8)

	def _get_agent_observation(self, n):
		""" Gets POMDP for each agent/node
		
		The POMDP is a vector depending on k slices with, a_k the flag to check if the task just arrived, b_k the buffer lenght on slice k, be_k the
		number of tasks under processing in the slice k and r_c and r_m the resources available in the node.

		Parameters:
			n: Fog_node - a node of the nodes list in the envrionment
		"""

		# does a partial system observation
		pobs = np.concatenate(([1 if len(n.buffers[k]) > 0 and self.clock == n.buffers[k][-1]._timestamp else 0 for k in range(n.max_k)],
			[len(n.buffers[k]) for k in range(n.max_k)], [n.being_processed_on_slice(k) for k in range(n.max_k)],
			[n._avail_cpu_units],[n._avail_ram_units]))
		return np.array(pobs, dtype=np.uint8)

	def _cap_action_n(self, action_n):
		""" Caps the action for a possible action to take. This way a model free algorithm will associte impossible actions with the capped version.

		Parameters:
			action_n: np.array - the action of the N agents/nodes to be taken in this timestep
		"""

		for n in range(len(action_n)):
			for i in range(len(action_n[n])):
				if action_n[n][i] >= self.action_space.nvec[n][i]:
					action_n[n][i] = self.action_space.nvec[n][i]-1
		return action_n

	def _set_agent_action(self, n, action):
		""" Sets an action from an agent/node n, translated in queueing events to the event queue.

		The action from the agent/node is given for each slice by, f_k meaning where the arrived task at the slice will be processed, if it's the same
		as the node index it'll be locally if it's different it'll be offloaded, and w_k meaning the number of tasks that this node will attempt to process
		on that slice.

		Parameters:
			n: Fog_node - a node of the nodes list in the envrionment
			action: np.arry - the action that was decided by the agent/node n
		"""

		# takes the action in the system, i.e. sets up the offloading events
		# for node n: [f_0, ..., f_k, w_0, ..., w_k]
		[fks, wks] = np.split(action, 2)
		# concurrent offloads
		concurr = sum([1 if fk!=n.index and fk!=0 else 0 for fk in fks])
		for k in range(DEFAULT_SLICES):
			# start processing if there is any request
			if wks[k] != 0:
				self.evq.add_event(Start_processing(self.clock, n, k, wks[k]))
			# and if you are given a destination, add the offload event
			if fks[k] != n.index and fks[k] != 0:
				self.evq.add_event(Offload_task(self.clock, n, k, self.nodes[fks[k]-1], concurr))

	def _agent_reward_fun(self, n, obs, action):
		""" Calculates the reward for an agent given his own observation and an action

		Parameters:
			n: Fog_node - a node of the nodes list in the envrionment
			obs: np.arry - the POMPD of the agent/node n
			action: np.arry - the action that was decided by the agent/node n
		"""

		# calculate the reward for the agent (node) n
		node_reward = 0
		[fks, wks] = np.split(action, 2)
		concurr = sum([1 if fk!=n.index and fk!=0 else 0 for fk in fks])
		for k in range(n.max_k):
			# only makes sense to calculate the coefficient if a[k] = 1
			if obs[k]:
				D_ik = 0; Dt_ik = 0
				# if it's offloaded adds communication time to delay
				if fks[k] != n.index and fks[k] != 0:
					bw = int(n.available_bandwidth()/concurr)
					if bw >= NODE_BANDWIDTH_UNIT: # just make sure it is actually going to offload
						Dt_ik = PACKET_SIZE / (point_to_point_transmission_rate(n._distances[fks[k]],concurr))
						D_ik += Dt_ik*1000 # converto to milliseconds
				# calculate the Queue delay: b_ik/service_rate_i
				D_ik += obs[n.max_k+k]/n._service_rate[k] # service rate per millisecond
				# and the processing delay T*slice_k_cpu_demand / CPU_UNIT (GHz)
				D_ik +=  1000* PACKET_SIZE*self.case["task_type"][k][1] / (CPU_UNIT) # convert it to milliseconds
				# finally, check if slice delay constraint is met
				if D_ik >= self.case["task_type"][k][0]:
					coeficient = -1
				else:
					coeficient = 1

				# count the number of new arrivals in the arriving node
				arr = 0
				for ev in self.evq.queue():
					if is_arrival_on_slice(ev, self.nodes[fks[k]-1], k) and ev.time <= self.clock+Dt_ik:
						arr += 1
				# also, verify if there is an overload chance in the arriving node
				arrival_node = self.nodes[fks[k]-1] if fks[k] > 0 else n
				arr_obs = self._get_agent_observation(arrival_node)
				if arr_obs[DEFAULT_SLICES+k]+arr+1 >= MAX_QUEUE:
					coeficient -= OVERLOAD_WEIGHT # tunable_weight

				# a_ik * ( (-1)if(delay_constraint_unmet) - (tunable_weight)if(overflow_chance) )
				node_reward += obs[k] * coeficient

		# float point reward: 1/k * reward on each slice
		return node_reward/n.max_k

	def render(self, mode='human', close=False):
		""" Renders the last step taken given a saved_step_info with the generic observation and the actions taken plus the current state (result) """

		# Render the environment to the screen
		if self.saved_step_info is None: return
		nodes_obs = self.saved_step_info[0]
		nodes_actions = self.saved_step_info[1]
		curr_obs = self._get_state_obs()
		print("------",round(self.clock*1000,2),"ms ------")
		for i in range(N_NODES):
			print("--- ",self.nodes[i]," ---")
			[a, b, be, rc, rm] = np.split(nodes_obs[i], [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
			print(" obs[ a:", a,"b:", b, "be:", be, "rc:",rc, "rm:",rm,"]")
			[f, w] = np.split(nodes_actions[i], 2)
			print("act[ f:",f,"w:",w,"]")
			[a, b, be, rc, rm] = np.split(curr_obs[i], [DEFAULT_SLICES, DEFAULT_SLICES*2, DEFAULT_SLICES*3, DEFAULT_SLICES*3+1])
			print(" obs[ a:", a,"b:", b, "be:", be, "rc:",rc, "rm:",rm,"]")
			#print("-- current state: --")
			#for k,buf in enumerate(self.nodes[i].buffers):
			#	print("slice",k,"buffer",[round(t._timestamp,4) for t in buf])
		for ev in self.evq.queue():
			if ev.classtype != "Task_arrival" and ev.classtype != "Task_finished":
				print(ev.classtype+"["+str(round(ev.time*1000,2))+"ms]", end='-->')
		print(round(1000*(self.clock+self.time_step)),"ms")
		input("\nEnter to continue...")
		pass

	def seed(self, seed=None):
		""" Seeds the envrionment setting a random state, the tools random state and returning the seed used """

		# set all the necessary seeds for reproductibility
		# one for the sim_env random requirements
		self.np_random, seed = seeding.np_random(seed)
		# and another for the global randoms
		set_tools_seed(seed)
		return [seed]

	def is_done(self):
		""" To verify if the envrionmnet is done """

		return self.clock >= self.max_time

# <<<<<
# >>>>> To map the action and observation spaces to a single integer if necessary
def map_int_vect_to_int(vect, maxes):
	retval = 0
	for v, m in zip(vect, maxes):
		assert v < m
		retval = retval*m + v
	return retval

def map_int_to_int_vect(num, maxes):
	vect = []; retval = num;
	for m in reversed(maxes):
		vect.insert(0,retval%m)
		retval = int(retval/m)
	return vect
# <<<<<