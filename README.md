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

