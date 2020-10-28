#!/usr/bin/env python

import numpy as np
import sys
if "/home/yourself/fog-computing-orchestration" not in sys.path:
    sys.path.append("/home/yourself/fog-computing-orchestration")

# functions under testing
from sim_env.calculators import db_to_linear, linear_to_db, channel_gain
from sim_env.calculators import shannon_hartley, euclidean_distance, bernoulli_arrival

def test_db_to_linear():
    # 10 * log10 ( value )
    assert db_to_linear(20) == 100
    assert db_to_linear(0) == 1
    assert db_to_linear(-20) == 0.01

def test_linear_to_db():
    # 10 ^ 0.1 * value
    assert linear_to_db(100) == 20
    assert linear_to_db(0.01) == -20
    assert linear_to_db(-100) == float("-inf")

def test_channel_gain():
    # linear_coef * distance ^ -exp_coef
    assert channel_gain(1,0.001,4) == 0.001
    assert channel_gain(2,1,2) == 0.25
    assert channel_gain(2,2,1) == 1
    assert channel_gain(0,0.001, 4) == 0

def test_euclidean_distance():
    # sqrt((x1-x2)^2 + (y1-y2)^2)
    assert euclidean_distance(0,0, 1, 0) == 1
    assert euclidean_distance(0,0, 1, 1) == np.sqrt(2)
    assert euclidean_distance(0,-1, 0, 1) == 2
    assert euclidean_distance(-1,0, -2, 0) == 1

def test_shannon_hartley():
    # r = B * log2(1 + P/N)
    assert shannon_hartley(0, 10,10,10) == 0
    assert shannon_hartley(1,7,1,1) == 3
    assert shannon_hartley(-100,1,1,1) == 0 # error
    assert shannon_hartley(100,1,-1,1) == 0 # error
    assert shannon_hartley(100,1,1,-20) == 0 # error
    assert shannon_hartley(1,-1,1,1) == 0 # error

def test_bernoulli_arrival():
    # True if x < p else False
    assert bernoulli_arrival(10) == 0 # error
    assert bernoulli_arrival(-1) == 0 # error
    count1 = 0; count0 = 0;
    for i in range(100000):
        if bernoulli_arrival(0.7):
            count1 +=1
        else:
            count0 +=1
    # ~50% of the times it hits, ~50% it doesn't
    assert round(count1/100000,1) == 0.7
    assert round(count0/100000,1) == 0.3
