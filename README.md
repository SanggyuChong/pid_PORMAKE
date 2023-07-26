# π-d PORMAKE

Python library for the construction of π-d conjugated metal-organic frameworks.

All credits to [Sangwon Lee](https://github.com/Sangwon91) for the development of original [PORMAKE](https://github.com/Sangwon91/PORMAKE) code. Here, a revised version of PORMAKE is presented with additional features to enforce coplanarity in placing the edges of the framework material, which is a common requirement in constructing π-d conjugated MOFs that commonly exhibit electrical conductivity.

## Installation
* Dependencies

```
python>=3.7
```

```
tensorflow>=1.15|tensorflow-gpu>=1.15
pymatgen<2022
ase>=3.18.0
```

1. Install all dependencies.

```bash
$ pip install -r requirements.txt
```

2. Install `pormake` using `setup.py`

```bash
$ python setup.py install
```
