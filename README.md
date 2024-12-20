# π-d PORMAKE

Python library for the construction of π-d conjugated metal-organic frameworks. Preprint for the associated work found [here](https://doi.org/10.26434/chemrxiv-2024-kzdm3).

All credits to [Sangwon Lee](https://github.com/Sangwon91) for the development of the original [PORMAKE](https://github.com/Sangwon91/PORMAKE) code.

This repository contains a revised version of PORMAKE with additional features to enforce co-planarity in placing the edges of the framework material, which is a common requirement in constructing π-d conjugated MOFs that can exhibit electrical conductivity.

Interested users are kindly asked to refer to the tutorials available in the original PORMAKE repository. For example cases in which planarity enforcement is performed for the construction of π-d conjugated frameworks (and also sample building block files prepared for the planarity enforcement), several Jupyter notebooks and building block `.xyz` files are available in the `notebooks` directory.


## Installation
* Dependencies

```
python>=3.7
```

```
tensorflow>=1.15
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
