# ShiftML

![Tests](https://img.shields.io/github/actions/workflow/status/lab-cosmo/shiftml/tests.yml?branch=main&logo=github&label=tests)

**Disclaimer: As with all machine learning models, ShiftML3 should be used within its domain of applicability and in a cautious manner.**

Welcome to ShiftML, a python package for the prediction of chemical shieldings of organic solids and beyond.

## Looking for quick chemical shielding predictions from your browser?

Please visit [shiftml.org](https://shiftml.org) !

## Usage

Use ShiftML with the atomistic simulation environment to obtain fast estimates of chemical shieldings:

```python

from ase.build import bulk
from shiftml.ase import ShiftML

frame = bulk("C", "diamond", a=3.566)
calculator = ShiftML("ShiftML3")

cs_iso = calculator.get_cs_iso(frame)
```


For more advanced predictions read also section [Advanced usage of the ShiftML3 model](#advanced-usage-of-the-shiftml3-model).

## Installation

This package is available on PyPI and can be installed using pip. The recommended way to install ShiftML is to use the following command:
**ShiftML supports Python 3.9–3.13.**

```
pip install shiftml
```

The recommended way to install ShiftML is to use a virtual environment, such as `venv` or `conda`, to avoid conflicts with other packages.

```
# Create a virtual environment
python -m venv shiftml-env

# Activate the virtual environment
source shiftml-env/bin/activate  # On Windows use: shiftml-env\Scripts\activate

# Install ShiftML
pip install shiftml

# source the environment in your script whenever you want to use ShiftML
```

### Known installation issues
The following installation issues are known:
- Old Intel-based Macs are not supported, because torch does not support them anymore (building torch binaries).
- We have switched recently the model engine from "metatensor.atomistic" to "metatomic". This is only a namespace issue and does not affect the models.
**If you receive a "Not a metatomic model" error message** you probably had an earlier ShiftML release installed and the old versions model files remain in cache.
In order to **clear the cache once**, please load the model once and overwrite the cache:

```python
calculator = ShiftML("ShiftML3", force_download=True)
```

## The code that makes it work

This project would not have been possible without the following packages:

- Metadata and model handling: [metatensor](https://github.com/metatensor/metatensor)
- Model trainings: [metatrain](https://github.com/metatensor/metatrain)


## Available models
The following models are available in ShiftML:
- **ShiftML3** : A model trained on a large dataset of chemical shieldings in organic solids, including anisotropy. It is trained on a dataset of 1.4 million chemical shieldings from 14000 organic crystals and can predict chemical shieldings for a wide range of organic solids. Containing at most the following 12 elements: H, C, N, O, S, F, P, Cl, Na, Ca, Mg and K. Against hold-out GIPAW-DFT data the model achieves isotropic shielding prediction accuracies (RMSE) of 0.43 ppm for $^{1}\text{H}$ and 2.32 ppm for $^{13}\text{C}$. [preprint](https://arxiv.org/abs/2506.13146)



## Advanced usage of the ShiftML3 model

The following section contains advanced usage examples of the ShiftML3 model,
which is currently the only supperted model used in the `ShiftML` calculator.

```python
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
cs_iso_uncertainty = np.std(cs_committee_iso, axis=1, ddof=1)

# Compute the chemical shielding anisotropy (from mean tensor prediction)

cs_psa = np.linalg.eigvalsh(cs_tensor)
```

This snippet will estimate the predicted chemical shieldings of diamond to be highly uncertain, 
as expected and desired, given that diamond as an inorganic material is not well 
represented in the training data of the model.


### Further usage options of the ShiftML calculator and ShiftML3 model

If you want to force the calculator to download model files again you can use the `force_download` argument:

```python
calculator = ShiftML("ShiftML3", force_download=True)
```

The model will look for the preferred device to run the model on (per default it will use the GPU if available, otherwise it will use the CPU). But you can also specify the device manually:

```python
calculator = ShiftML("ShiftML3", device="cpu")  # run always on CPU

calculator = ShiftML("ShiftML3", device="cuda")  # run always on GPU
```

## Help us improve ShiftML
If you find bugs or have suggestions for improvements, please open an issue on the [ShiftML GitHub repository](https://github.com/lab-cosmo/ShiftML/issues).
Do you have systems for which you find that the model does not work well? - please let us know on github, or email us. We are more than happy to hear from you, and if you provide us with the systems, we can try to improve the model in the future.

Are you missing chemical elements for which you would like to have chemical shielding predictions, or your systems that contain elements that are not supported by the current model? Please let us know, so we can consider adding them in the future.

## Reproducibility

To ensure reproducibility of shielding predictions with ShiftML, you can save the pipy package version of the ShiftML package you used. This can be done by running the following command in your terminal (assuming you have ShiftML installed in your current Python environment):

```bash
pip freeze | grep shiftml > shiftml_version.txt
```

Then, if you want to reproduce the results, you can install the exact version of ShiftML that you used by running, or simply specifying the version in the pip install command:

```bash
pip install -r shiftml_version.txt

# or

pip install shiftml==<version>
```

## FAQ
### ShiftML3 – Frequently Asked Questions


<details>
<summary><strong>ShiftML3 predictions aren’t identical for magnetically equivalent atoms. Why?</strong></summary>

ShiftML3 is built on the **Point Edge Transformer (PET)** model, which is *not perfectly rotationally invariant*.  
This can introduce tiny, random differences for atoms that are magnetically equivalent.  
We have verified that these fluctuations are minor and do **not** harm overall accuracy.

> **Tip – get identical shielding predictions**  
> Average the predictions over all magnetically equivalent atoms.

</details>

---

<details>
<summary><strong>ShiftML3 shows large errors versus my GIPAW-DFT shieldings. What’s going on?</strong></summary>

Chemical-shielding calculations are *very* sensitive to the **code and convergence parameters** used.  
Only compare ShiftML3 to GIPAW-DFT data generated with *exactly* the same settings as the training set.

*Reference inputs* for Quantum Espresso with the correct parameters are available in this  
[Zenodo data repository](https://zenodo.org/records/7097427).

</details>

---

<details>
<summary><strong>I used identical GIPAW-DFT parameters but still see big errors. What now?</strong></summary>

Check the model’s **uncertainty estimates** (committee variance; see “Advanced usage” above).  
If the uncertainty is **several ×** the element’s test-set RMSE, the prediction is probably unreliable
for your structure.

</details>

---

<details>
<summary><strong>My calculated shieldings don’t correlate with experiment at all. Why?</strong></summary>

1. **Validate the baseline.**  
   Make sure reliable **GIPAW/PBE** results exist (or recompute them) and confirm they correlate with experiment.  
   Inaccurate DFT—often the exchange–correlation functional—can be blamed.

2. **Check your structures.**  
   If candidate geometries don’t reflect experimental conditions *or* the inter-atomic potential used to generate structures is poor,
   both DFT and ML predictions will stray from reality.

</details>

### Installation in a virtual environment from source
It is highly recommended to install ShiftML in a virtual environment to avoid conflicts with other packages. You can use `venv` or `virtualenv` to create a virtual environment.

```bash
python -m venv shiftml_env
source shiftml_env/bin/activate  # On Windows use `shiftml_env\Scripts\activate`
git clone https://github.com/lab-cosmo/ShiftML.git
cd ShiftML
pip install .
```

### Installation with conda from source
If you prefer to use conda, you can create a new environment and install ShiftML there. This is especially useful if you want to manage dependencies more easily.

```bash
conda create -n shiftml python=3.12
conda activate shiftml
git clone https://github.com/lab-cosmo/ShiftML.git
cd ShiftML
pip install .
```

### Verify the installation
To verify that ShiftML is working as intended you can run the regressiontests provided in the package. This will ensure that the installation was successful and that the package is functioning correctly.
To run the test, install pytest in your python environment:

```bash
pip install pytest
```

Then run the tests, by changing into the tests directory and running pytest:

```bash
cd tests
pytest
```

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
- A deep learning model for chemical shieldings in molecular organic solids including anisotropy - Kellner, Holmes, Rodriguez Madrid, Viscosi, Zhang, Emsley, Ceriotti  [[4](https://doi.org/10.1021/acs.jpclett.5c01819)]

