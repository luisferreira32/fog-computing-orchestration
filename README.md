# fog-computing-orchestration

The objective of this work is to work on the state-of-the-art for Fog Computing (FC) orchestration/choreography. It is envisioned to propel our society to a new era of hyper-information.


---


## Technical details

To run this simulations it is necessary to have Tensorflow 2.0 installed, recommended to create a venv in the same folder as the source code:
[Official TF2 installation guide](https://www.tensorflow.org/install/pip#virtual-environment-install) (link accessed 08/2020)

[Pytest](https://docs.pytest.org/en/stable/getting-started.html) is also an excellent tool for debuging the application

[Matplotlib](https://matplotlib.org/) was used for graphical results

### Installed package list  
<pre>
Package                  Version
------------------------ ---------
absl-py                  0.9.0
astunparse               1.6.3
atomicwrites             1.4.0
attrs                    19.3.0
cachetools               4.1.1
certifi                  2020.6.20
chardet                  3.0.4
colorama                 0.4.3
cycler                   0.10.0
gast                     0.3.3
google-auth              1.20.0
google-auth-oauthlib     0.4.1
google-pasta             0.2.0
grpcio                   1.30.0
h5py                     2.10.0
idna                     2.10
iniconfig                1.0.1
Keras-Preprocessing      1.1.2
kiwisolver               1.2.0
Markdown                 3.2.2
matplotlib               3.3.0
more-itertools           8.4.0
numpy                    1.18.5
oauthlib                 3.1.0
opt-einsum               3.3.0
packaging                20.4
Pillow                   7.2.0
pip                      20.2.1
pluggy                   0.13.1
protobuf                 3.12.4
py                       1.9.0
pyasn1                   0.4.8
pyasn1-modules           0.2.8
pyparsing                2.4.7
pytest                   6.0.1
python-dateutil          2.8.1
requests                 2.24.0
requests-oauthlib        1.3.0
rsa                      4.6
scipy                    1.4.1
setuptools               47.1.0
six                      1.15.0
tensorboard              2.3.0
tensorboard-plugin-wit   1.7.0
tensorflow               2.3.0
tensorflow-estimator     2.3.0
tensorflow-gpu           2.3.0
tensorflow-gpu-estimator 2.3.0
termcolor                1.1.0
toml                     0.10.1
urllib3                  1.25.10
Werkzeug                 1.0.1
wheel                    0.34.2
wrapt                    1.12.1
</pre>

### For simulator

When using fog objects, import the configurations that can give you a higher level tweaking ability
<pre> import fog.configs </pre>