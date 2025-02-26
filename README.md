# ShiftML

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://lab-cosmo.github.io/ShiftML/latest/)
![Tests](https://img.shields.io/github/check-runs/lab-cosmo/ShiftML/main?logo=github&label=tests)

**Disclaimer: This package is still under development and should be used with caution.**

Welcome to ShitML, a python package for the prediction of chemical shieldings of organic solids and beyond.



## Usage

Use ShiftML with the atomsitic simulation environment to obtain fast estimates of chemical shieldings:

```python

from ase.build import bulk
from shiftml.ase import ShiftML

frame = bulk("C", "diamond", a=3.566)
calculator = ShiftML("ShiftML2.1dev_ensemble")

cs_iso = calculator.get_cs_iso(frame)

print(cs_iso)

```



## Installation

To install ShiftML, you can use clone this repository and install it using pip, a pipy release will follow soon:

```
pip install .[nanopet]
```


## IMPORTANT: Install pre-instructions before PiPy release

Featomic-torch, one of the main dependence of ShiftML models that require Featomi, requires CXX and Rust compilers to be built from source.
Most systems come already with configured C/C++ compilers (make sure that some environment variables CC and CXX are set
and gcc can be found), but Rust typically needs to be installed manually.
For ease of use we strongly recommend to use some sort of package manager to install Rust, such as conda and a fresh environment.


```bash

conda create -n shiftml python=3.12
conda activate shiftml
conda install -c conda-forge rust

```

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu .[featomic]
```

## The code that makes it work

This project would not have been possible without the following packages:

- Metadata and model handling: [metatensor](https://github.com/metatensor/metatensor)
- Atomic descriptor engine: [featomic](https://github.com/metatensor/featomic)

## Documentation

The documentation is available [here](https://lab-cosmo.github.io/ShiftML/latest/).

## Contributors

Matthias Kellner\
Yuxuan Zhang\
Ruben Rodriguez Madrid\
Guillaume Fraux

## References

This package is based on the following papers:

- Chemical shifts in molecular solids by machine learning - Paruzzo et al. [[1](https://doi.org/10.1038%2Fs41467-018-06972-x)]
- A Bayesian approach to NMR crystal structure determination - Engel et al. [[2](https://doi.org/10.1039%2Fc9cp04489b)]
- A Machine Learning Model of Chemical Shifts for Chemically and\
Structurally Diverse Molecular Solids - Cordova et al. [[3](https://doi.org/10.1021/acs.jpcc.2c03854)]

