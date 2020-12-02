# fog-computing-orchestration

The objective of this work is to work on the state-of-the-art for Fog Computing (FC) orchestration/choreography. It is envisioned to propel our society to a new era of hyper-information.


---

##  How to run

First install the necessary dependencies - Tensorflow 2.x and Matplotlib for results.

For help in commands run the command:
```
python playground.py --help
```
If you want to run all cases defined in `sim_env/configs.py`, with the baseline algorithms (Nearest Round Robin or Nearest Priority Queue):
```
python playground.py --algorithm=rrpq --cases=all
```
To train a Reinforcement Learning, for example Advantage Actor Critc agent in a case described in `sim_env/configs.py`, use: [NOTE: still bugged]
```
python playground.py --algorithm=a2c --cases=n1 --train
```

---
## Technical details
  
DISCLAIMER: each of this libraries is not mine, the authors are as stated in the links below, and they were only used in this work for academic purposes.  

To run this simulations it is necessary to have Tensorflow 2.x installed and its dependencies, recommended to create a venv in the same folder as the source code:
[Official TF2 installation guide](https://www.tensorflow.org/install/pip#virtual-environment-install) (link accessed 08/2020)  
  
[Pytest](https://docs.pytest.org/en/stable/getting-started.html) was a tool used for debugging and testing the scripts.  
  
[Matplotlib](https://matplotlib.org/) was used for graphical results.


