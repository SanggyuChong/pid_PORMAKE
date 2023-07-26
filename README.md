# pi-d PORMAKE
**Por**ous materials **Make**r

> Python library for the construction of pi-d conjugated metal-organic frameworks.

All credits to Sangwon Lee for the development of original PORMAKE code. Here, a revised version of PORMAKE is presented with additional features to enforce coplanarity for the edges, which is a common requirement in constructing pi-d conjugated MOFs.

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
