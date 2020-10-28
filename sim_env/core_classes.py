#!/usr/bin/env python

'''
TODO@luis: check this list and implement stuff
class Cloud_node
'''

# external imports
from abc import ABC, abstractmethod
from collections import deque
from numpy import random

# sim_env imports
from sim_env.configs import MAX_QUEUE, CPU_UNIT, RAM_UNIT, CPU_CLOCKS, RAM_SIZES
from sim_env.configs import AREA, PATH_LOSS_CONSTANT, PATH_LOSS_EXPONENT, THERMAL_NOISE_DENSITY
from sim_env.configs import NODE_BANDWIDTH, TRANSMISSION_POWER
from sim_env.calculators import euclidean_distance, channel_gain, shannon_hartley, db_to_linear

# ---------- Nodes related classes and functions ---------

class Node(ABC):
    """ Abstract node class
    """
    def __init__(self, index, x, y, cpu_frequency, ram_size, number_of_slices):
        super().__init__()
        # Identifiers
        self.index = index
        self.name = "node_"+str(index)
        # placement
        self.x = x
        self.y = y
        # resources
        self.cpu_frequency = cpu_frequency
        self.ram_size = ram_size
        self._avail_cpu_frequency = cpu_frequency
        self._avail_ram_size = ram_size
        # slices buffers
        self.max_k = number_of_slices
        self.buffers = [deque(maxlen=MAX_QUEUE) for _ in range(number_of_slices)]

    def __str__(slef):
        return self.name

    @abstractmethod
    def add_task_on_slice(self, k, task):
        # returns the task if the slice buffer is full
        pass
    
    @abstractmethod
    def remove_task_of_slice(self, k, task):
        # removes and returns a task if it is or not on the buffer
        pass

    @abstractmethod
    def start_processing_in_slice(self, k, w):
        # starts processing w tasks in slice k
        pass


class Fog_node(Node):
    """ A Fog node with limited resources
    """
    def __init__(self, index, x, y, cpu_frequency, ram_size, number_of_slices):
        super(Fog_node, self).__init__(index, x, y, cpu_frequency, ram_size, number_of_slices)

    def slice_buffer_len(self, k):
        # error shield
        if not (0<=k<self.max_k): return 0
        return len(self.buffers[k])

    def add_task_on_slice(self, k, task):
        # error shield
        if not (0<=k<self.max_k): return task
        # returns the task if the slice buffer is full
        if len(self.buffers[k]) == self.buffers[k].maxlen:
            return task
        self.buffers[k].append(task)
        return None
    
    def remove_task_of_slice(self, k, task, finish_time=None):
        # error shield
        if not (0<=k<self.max_k): return None
        # removes and returns a task if it is or not on the buffer
        try:
            self.buffers[k].remove(task)
            # values should be zero if it's not processing
            self._avail_ram_size += task._memory_units*RAM_UNIT
            self._avail_cpu_frequency += task._cpu_units*CPU_UNIT
            # and delay will still be -1 if finish_time is default
            if finish_time==None: finish_time = task._timestamp-1
            task.finish_processing(finish_time)
        except:
            return None
        return task

    def start_processing_in_slice(self, k, w):
        # error shield
        if w <= 0 or not (0<=k<self.max_k): return
        # starts w tasks in slice k, within available resources constraints
        aux_w = w
        # only process if there is a task on the buffer
        for task in self.buffers[k]:
            # only process if has cores and memory for it
            if not task.is_processing() and aux_w > 0 and self._avail_cpu_frequency > 0 and self._avail_ram_size >= task.ram_demand:
                # calculate units to attribute to this task
                n_cpu_units = int(self._avail_cpu_frequency/aux_w)
                if n_cpu_units == 0:
                    n_cpu_units = 1
                n_memory_units = int(task.ram_demand/RAM_UNIT)
                # and take them from the available pool
                self._avail_cpu_frequency -= n_cpu_units*CPU_UNIT
                self._avail_ram_size -= n_memory_units*RAM_UNIT
                # then start the processing
                task.start_processing(n_cpu_units, n_memory_units)
                # reduce the number that we will still allocate
                aux_w -= 1


def point_to_point_transmission_rate(n1, n2):
    # calculates transmission rate given two nodes
    d = euclidean_distance(n1.x, n1.y, n2.x, n2.y)
    g = channel_gain(d, PATH_LOSS_CONSTANT, PATH_LOSS_EXPONENT)
    p_mw = db_to_linear(TRANSMISSION_POWER)
    n0_mw = db_to_linear(THERMAL_NOISE_DENSITY)
    return shannon_hartley(g, p_mw, NODE_BANDWIDTH, n0_mw)

def create_random_node(index=0):
    # returns a node uniformly sampled within configurations, after the previous index
    [x, y] = [random.randint(low=0, high=AREA[0]), random.randint(low=0, high=AREA[1])]
    number_of_slices = 3
    cpu = random.choice(CPU_CLOCKS)
    ram = random.choice(RAM_SIZES)
    return Fog_node(index, x, y, cpu, ram, number_of_slices)


# ---------- Task related classes and functions ---------

class Task():
    """ A task attributed by the users
    """
    def __init__(self, timestamp, packet_size, delay_constraint, cpu_demand, ram_demand):
        self.packet_size = packet_size
        self.delay_constraint = delay_constraint
        self.cpu_demand = cpu_demand
        self.ram_demand = ram_demand

        self._processing = False
        self._memory_units = 0
        self._cpu_units = 0
        self._timestamp = timestamp
        self._total_delay = -1
        self._expected_delay = -1

    def __str__(self):
        return str(self._timestamp)+"s is_processing "+str(self._processing)

    def is_processing(self):
        return self._processing

    def is_completed(self):
        return False if self._total_delay == -1 else True

    def start_processing(self, cpu_units, memory_units):
        self._processing = True
        self._cpu_units = cpu_units
        self._memory_units = memory_units
        self._expected_delay = task_processing_time(self)

    def finish_processing(self, finish_time):
        self._processing = False
        self._total_delay = finish_time-self._timestamp

    def task_delay(self):
        return self._total_delay

def task_processing_time(t=None):
    # calculates the time a task takes to process after starting to process
    if t is None: return "no task given"
    if not t.is_processing(): return "task not processing"
    # task has cpu_demand cycles/bit (packet size in Mb)
    total_cycles = t.cpu_demand*t.packet_size*10**6
    # processing units are in 1GHz each
    total_time = total_cycles/(t._cpu_units*CPU_UNIT*(10**9))
    return total_time

def task_communication_time(t, bit_rate):
    if t is None: return "no task given"
    # simple packet_size calc
    return t.packet_size/bit_rate