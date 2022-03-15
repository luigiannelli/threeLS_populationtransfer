# A Tutorial on Optimal Control and Reinforcement Learning methods for Quantum Technologies
by Luigi Giannelli, Pierpaolo Sgroi, Jonathon Brown, Gheorghe Sorin Paraoanu, Mauro Paternostro, Elisabetta Paladino, and Giuseppe Falci  
[Physics Letters A 434, 128054 (2022)](https://doi.org/10.1016/j.physleta.2022.128054)  
[arXiv:2112.07453](https://arxiv.org/abs/2112.07453)

## Installation
Tested with *python 3.9.7*.

1. Clone the repository
``` shell
git clone https://www.github.com/luigiannelli/threeLS_populationTransfer
cd threeLS_populationTransfer
```

2. (Optionally) Create a python environment (instructions using
   [pyenv](https://github.com/pyenv/pyenv) with
   [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv))
``` shell
pyenv install 3.9.7
pyenv local 3.9.7
pyenv virtualenv threeLS
pyenv local threeLS
```

3. Install dependencies
``` shell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Run [jupyterlab](https://jupyter.org/) and play with the notebooks in
   `notebooks`

``` shell
jupyterlab
```
