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
calculator = ShiftML("ShiftML3")

cs_iso = calculator.get_cs_iso(frame)
```

For more advanced predictions read also section [Advanced usage of the ShiftML3 model](#advanced-usage-of-the-shiftml3-model).

## Installation

To install ShiftML, you can use clone this repository and install it using pip, a pipy release will follow soon. 

```
pip install git+https://github.com/lab-cosmo/ShiftML.git
```

### Installation in a virtual environment
It is highly recommended to install ShiftML in a virtual environment to avoid conflicts with other packages. You can use `venv` or `virtualenv` to create a virtual environment.

```bash
python -m venv shiftml_env
source shiftml_env/bin/activate  # On Windows use `shiftml_env\Scripts\activate`
pip install .
```

### Installation with conda
If you prefer to use conda, you can create a new environment and install ShiftML there. This is especially useful if you want to manage dependencies more easily.

```bash
conda create -n shiftml python=3.12
conda activate shiftml
pip install .
```

### Known installation issues
The following installation issues are known:
- Old Intel-based Macs are not supported, because torch does not support them anymore (building torch binaries).

## The code that makes it work

This project would not have been possible without the following packages:

- Metadata and model handling: [metatensor](https://github.com/metatensor/metatensor)
- Model trainings: [metatrain](https://github.com/metatensor/metatrain)


## Available models
The following models are available in ShiftML:
- **ShiftML3** : A model trained on a large dataset of chemical shieldings in organic solids, including anisotropy. It is trained on a dataset of 1.4 million chemical shieldings from 14000 organic crystals and can predict chemical shieldings for a wide range of organic solids. Containing at most the following 12 elements: H, C, N, O, S, F, P, Cl, Na, Ca, Mg and K. Against hold-out GIPAW-DFT data the model achieves isotropic shielding prediction accuracies (RMSE) of 0.43 ppm for $^{1}\text{H}$ and 2.32 ppm for $^{13}\text{C}$. 


## Advanced usage of the ShiftML3 model

The following section contains advanced usage examples of the ShiftML3 model,
which is currently the only supperted model used in the `ShiftML` calculator.

```
from ase.build import bulk
from shiftml.ase import ShiftML
import numpy as np

frame = bulk("C", "diamond", a=3.566)
calculator = ShiftML("ShiftML3")

# Get isotropic chemical shieldings
cs_iso = calculator.get_cs_iso(frame)

# Get the symmetric tensor of chemical shieldings
cs_tensor = calculator.get_cs_tensor(frame)

# Get the full chemical shielding tensor (including antisymmetric components)
cs_full_tensor = calculator.get_cs_tensor(frame, return_symmetric=False)

# Get the committe predictions:
cs_committee_iso = calculator.get_cs_iso_ensemble(frame)
cs_committee_tensor = calculator.get_cs_tensor_ensemble(frame)

# Compute uncertainty estimates for the isotropic chemical shieldings
cs_iso_uncertainty = np.std(cs_committee_iso, axis=1)

# Compute the chemical shielding anisotropy
# (from the committee predictions): compute committee eigenvalues first and then average

```

This snippet will estimate the predicted chemical shieldings of diamond to be highly uncertain, 
as expected and desired, given that diamond as an inorganic material is not well 
represented in the training data of the model.

## FAQ
- ShiftML3 predictions are not exactly identical between magnetically equivalents, why?
    - ShiftML3 is based on the Point Edge Transformer - PET model, which does not makr exactly rotationally invariant predictions. This means that the model can make slightly different predictions for magnetically equivalent atoms. We have carefully tested, that the rotational fluctuations are small and do not affect the overall performance of the model. We recommend averaging the predictions over magnetically equivalent atoms to obtain identical predictions for equivalent atoms.
- ShiftML3 makes large prediction errors against my GIPAW-DFT shielding data, why?
    - Be aware that chemical shielding computations are very sensitive to the choice of convergence parameters and code used. You should only compare ShiftML3 predictions against GIPAW-DFT data computed with the same code and convergence parameters as used in the training of the model. You can find input files for Quantum Espresso GIPAW calculations with the same parameters as used in the training of ShiftML3 in this [data repository](https://zenodo.org/records/7097427)
- I am using the same GIPAW-DFT parameters as used in the training of ShiftML3, but the model still makes large prediction errors, why?
    - Check the uncertainty estimates of the model, which are computed from the committee predictions (see how it can be done in the advanced usage section above). If the uncertainty is large (especially when it is multiple times the test set RMSE of the given element of ShiftML3), the model is likely not able to make a reliable prediction for your system.


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
- A deep learning model for chemical shieldings in molecular organic solids including anisotropy - Kellner, Holmes, Rodriguez Madrid, Viscosi, Zhang, Emsley, Ceriotti  (in preparation)

