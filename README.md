# fog-computing-orchestration

The objective of this work is to work on the state-of-the-art for Fog Computing (FC) orchestration/choreography. It is envisioned to propel our society to a new era of hyper-information.


---
## Technical details
  
DISCLAIMER: each of this libraries is not mine, the authors are as stated in the links below, and they were only used in this work for academic purposes.  

To run this simulations it is necessary to have Tensorflow 2.x installed and its dependencies, recommended to create a venv in the same folder as the source code:
[Official TF2 installation guide](https://www.tensorflow.org/install/pip#virtual-environment-install)

[//]: # ([Tensorflow probability] https://www.tensorflow.org/probability is also one of the packages used.)
  
[Pytest](https://docs.pytest.org/en/stable/getting-started.html) was a tool used for debugging and testing the scripts.  
  
[Matplotlib](https://matplotlib.org/) was used for graphical results.

---
## How to run...  

First get the libraries and their dependencies, recommended to use a virtual environment (as shown in the Official TF2 installation guide).  

To run basic algorithms for results, you can use functions from `algorithms.runners` which follow the recommended OpenAI structure for test run. This functions also collect some information with indicators, and run accordingly to the algorithms given from `algorithms.basic` and in a `sim_env.environment.Fog_env` object environment created with the configurations in `sim_env.configs`.  
  
A basic run with an algorithm that schedules according to a Priority Queue method and offloads to the Nearest Node if a threshold is reached,
```python
from algorithms.runners import run_basic_algorithm_on_envrionment
from algorithms.basic import Nearest_Priority_Queue
from sim_env.environment import Fog_env
from utils.display import info_gather_init

env = Fog_env()
# this run will display some results in the command line, you can pick them up from the return values too
run_basic_algorithm_on_envrionment(Nearest_Priority_Queue, env, case, info_gather_init(), debug=False)
```
  
Note that the cases described in `sim_env.configurations` that will be assigned to the environment, must match the variables in that file. For example, if the number of slices in the Fog is set to 3, you cannot run a case that only describes one slice.  
  
To run a reinforcement learning algorithm, and to train it, you can use the function given in `algorithms.runners` or create a script on your own. Changing the way an algorithm trains is done on the algorithms' file itself. Most algorithms hyperparameters are defined in `algorithms.configs` and can be changed there too. For example, if you want to train the Advantage Actor-Critic algorithm created in `algorithms.a2c`,   
```python
from algorithms.runners import run_rl_algorithm_on_envrionment
from algorithms.a2c import A2c_Orchestrator
from sim_env.environment import Fog_env
from utils.display import info_gather_init

env = Fog_env()
# this run will display some results in the command line, you can pick them up from the return values too
run_rl_algorithm_on_envrionment(Nearest_Priority_Queue, env, case, info_gather_init(), debug=False, train=True, save=False, load=False)
```
  
A reinforcement learning algorithm purpose is to maximize a reward given by the environment, those rewards are defined in `sim_env.rewards` and can be then imported in the `sim_env.environment` to be used as the reward function of the environment itself.  

---
## List of possible improvements
### on the environment package sim_env...
- Re-model channel gain with friis equation;
- Add node ABC class;
- Implement Cloud node, subclass of node;  
- Re-model communications (in general) for more options of architectures;
- ... 

### on the orchestrator algorithms...
- Create orchestrator ABC class;
- ...

