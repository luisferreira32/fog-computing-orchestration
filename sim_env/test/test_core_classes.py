#!/usr/bin/env python

import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
    sys.path.append("/home/yourself/fog-computing-orchestration")

# imported test classes
from sim_env.configs import MAX_QUEUE
from sim_env.core_classes import Task, Fog_node

# ---------- test Task class methods ----------

def test_task_creation():
    t = Task(0, 5, 10, 400, 400)
    assert t.packet_size == 5
    assert t._timestamp == 0
    assert t._processing == False
    assert t._cpu_units == 0
    assert t._memory_units == 0
    assert t._total_delay == -1

def test_task_processing():
    t = Task(0, 5, 10, 400, 400)
    assert t.is_completed() == False
    assert t.is_processing() == False
    t.start_processing(1,1)
    assert t.is_processing() == True
    t.finish_processing(1)
    assert t.is_processing() == False
    assert t.is_completed() == True
    assert t.task_delay() == 1

# ---------- test Fog_node class methods ----------

def test_node_creation():
    # placement = [0,0], CPU = 1GHz, RAM = 2400MB
    f1 = Fog_node(0, x=0, y=0, cpu_frequency=1, ram_size=2400, number_of_slices=3)
    assert f1.index == 0
    assert f1.x == 0
    assert f1.y == 0
    assert f1.cpu_frequency == 1
    assert f1.ram_size == 2400
    assert len(f1.buffers) == 3

def test_node_task_on_slice():  
    f1 = Fog_node(0, x=0, y=0, cpu_frequency=1, ram_size=2400, number_of_slices=3)
    # timestamp=0s, size=5* 10^6bits, max_delay=10ms, cpu=400cycles/bit, ram=400MB
    t = Task(0, 5, 10, 400, 400)
    assert f1.add_task_on_slice(0,t)==None
    assert f1.remove_task_of_slice(0,t) == t
    assert f1.remove_task_of_slice(0,t) == None
    # now get full Queue and try to add one more
    for i in range(MAX_QUEUE):
        assert f1.add_task_on_slice(0,t) == None
    assert f1.add_task_on_slice(0,t) == t
   
def test_node_processing_on_slice():
   # try to process 2 task with capacity just for one
    f1 = Fog_node(0, x=0, y=0, cpu_frequency=1, ram_size=2400, number_of_slices=3)
    t1 = Task(0, 5, 10, 400, 400)
    t2 = Task(0,5,10,400,400)
    assert f1.add_task_on_slice(0,t1) == None
    assert f1.add_task_on_slice(0,t2) == None
    assert f1._avail_cpu_frequency == 1
    f1.start_processing_in_slice(0,2)
    assert f1._avail_cpu_frequency == 0
    assert f1.buffers[0][0].is_processing() # first started
    assert f1.buffers[0][1].is_processing() == False # second didn't
    assert f1.remove_task_of_slice(0, t1) == t1
    assert f1._avail_cpu_frequency == 1
    assert t1.is_processing() == False
    assert t1.is_completed() == False # since we just removed and didn't set finish time
    
def test_node_processing_on_slice_2():
    # limit the memory now
    f2 = Fog_node(0, x=0, y=0, cpu_frequency=10, ram_size=2400, number_of_slices=3)
    t1 = Task(0, 5, 10, 400, 400)
    t2 = Task(0,5,10,400,400)
    t3 = Task(1,5,10,400,1200)
    t4 = Task(1,5,10,400,1200)
    f2.add_task_on_slice(0,t1)
    f2.add_task_on_slice(0,t2)
    f2.add_task_on_slice(0,t3)
    f2.add_task_on_slice(0,t4)
    assert f2._avail_cpu_frequency == 10
    f2.start_processing_in_slice(0,4)
    assert f2._avail_cpu_frequency == 3


