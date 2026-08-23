# ShiftML

![Tests](https://img.shields.io/github/actions/workflow/status/lab-cosmo/shiftml/tests.yml?branch=main&logo=github&label=tests)

**Disclaimer: As with all machine learning models, ShiftML models should be used within its domain of applicability and in a cautious manner.**

Welcome to ShiftML, a python package for the prediction of chemical shieldings of organic solids and beyond.

## Looking for quick chemical shielding predictions from your browser?

Please visit [shiftml.org](https://shiftml.org) !

## Usage

Use ShiftML with the atomistic simulation environment to obtain fast estimates of chemical shieldings:

```python

from ase.build import bulk
from shiftml.ase import ShiftML

frame = bulk("C", "diamond", a=3.566)
calculator = ShiftML("ShiftML4")

cs_iso = calculator.get_cs_iso(frame)
```


For more advanced predictions read also section [Advanced usage of the ShiftML models](#advanced-usage-of-the-shiftml-models).

## Installation

This package is available on PyPI and can be installed using pip. The recommended way to install ShiftML is to use the following command:
**ShiftML supports Python 3.10–3.13.**

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
calculator = ShiftML("ShiftML4", force_download=True)
```

## Using ShiftML together with metatrain, PET-MAD or uPET

ShiftML deliberately keeps its dependency surface as small as possible, so that
it can live in the same environment as other packages of the metatensor
ecosystem:

- it depends on `metatensor-torch`, `metatomic-torch` and `metatomic-ase`, and
  tracks the versions used by recent `metatrain` releases;
- it does **not** depend on `metatensor-operations` or `metatensor-learn`.
  Those are only needed to *train* models; everything the ShiftML models need
  from them is already compiled into the published TorchScript files.

The model files published on Zenodo were exported against an older
`metatensor-torch`. ShiftML rebuilds their metatomic wrapper every time they are
loaded (see [Model files and metatomic versions](#model-files-and-metatomic-versions)),
so you never have to worry about which release exported them.

## Model files and metatomic versions

Each ShiftML model is distributed as a set of TorchScript archives, one per
committee member: eight for ShiftML3, and seven for ShiftML4 (its Zenodo record
holds eight files, but `model_0` is a duplicate upload of `model_1` and is
skipped, so that member is not counted twice). Every archive contains the
scripted network *and* a thin `AtomisticModel` wrapper around it, built by
whichever `metatomic-torch` was installed when the model was exported.

The network ages well — every TorchScript class and operator it uses keeps a
compatible schema. The wrapper does not: recent `metatomic-torch` expects
attributes on it that did not exist when the published models were written, and
`metatomic.torch.load_atomistic_model()` therefore refuses to open them.

So ShiftML never uses the wrapper that came with the file. On every load it
takes the network out and builds a fresh wrapper with the installed metatomic:

1. the archive is opened with `torch.jit.load`, which still works;
2. the scripted network is taken out of the old wrapper *as-is* — no weight is
   read, converted or re-initialised;
3. the neighbor-list request that lived on the old wrapper is re-attached;
4. a fresh `metatomic.torch.AtomisticModel` is built around it.

This happens inside `ShiftML(...)`, once per committee member, and costs about
0.1 s each. The model file on disk is never modified, and predictions are
exactly the ones the file encodes.

It is deliberately unconditional rather than a fix-up for old files. Any future
change to `AtomisticModel` breaks every previously exported model, however
recently it was written, and re-wrapping is the fix in those cases too — so
ShiftML keeps working across metatomic releases without needing new model files
or a new release of its own.

If you want a model file that other engines can open directly — LAMMPS, i-PI, or
plain metatomic — save the re-wrapped model:

```python
from shiftml.utils.loading import load_model

load_model("model_0.pt").save("model_0_for_current_metatomic.pt")
```

## The code that makes it work

This project would not have been possible without the following packages:

- Metadata and model handling: [metatensor](https://github.com/metatensor/metatensor)
- Model trainings: [metatrain](https://github.com/metatensor/metatrain)


## Available models
The following models are available in ShiftML:
- **ShiftML3** : A model trained on a large dataset of chemical shieldings in organic solids, including anisotropy. It is trained on a dataset of 1.4 million chemical shieldings from 14000 organic crystals and can predict chemical shieldings for a wide range of organic solids. Containing at most the following 12 elements: H, C, N, O, S, F, P, Cl, Na, Ca, Mg and K. Against hold-out GIPAW-DFT data the model achieves isotropic shielding prediction accuracies (RMSE) of 0.43 ppm for $^{1}\text{H}$ and 2.32 ppm for $^{13}\text{C}$. [preprint](https://arxiv.org/abs/2506.13146). Select the model as `ShiftML("ShiftML3")` in the ASE calculator.
- **ShiftML4** : A model trained on a large dataset of chemical shieldings in organic solids, including anisotropy, on PBE0 molecular corrected GIPAW-PBE data. It is trained on a dataset of 1.2 million chemical shieldings from 12600 organic crystals and can predict chemical shieldings for a wide range of organic solids. Containing at most the following 12 elements: H, C, N, O, S, F, P, Cl, Na, Ca, Mg and K. Against hold-out PBE0-molecular corrected GIPAW-DFT data the model achieves isotropic shielding prediction accuracies (RMSE) of 0.40 ppm for $^{1}\text{H}$ and 2.22 ppm for $^{13}\text{C}$, compared to 0.42 ppm and 2.24 ppm for ShiftML3, of the same hold-out set computed at the GIPAW-PBE data. Select the model as `ShiftML("ShiftML4")` in the ASE calculator.



## Advanced usage of the ShiftML models

The following section contains advanced usage examples of the ShiftML4 model,
which is currently one of the two supported models used in the `ShiftML` calculator.

```python
from ase.build import bulk
from shiftml.ase import ShiftML
import numpy as np

frame = bulk("C", "diamond", a=3.566)
calculator = ShiftML("ShiftML4")

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


### Further usage options of the ShiftML calculator and ShiftML3/ShiftML4 models.

If you want to force the calculator to download model files again you can use the `force_download` argument:

```python
calculator = ShiftML("ShiftML4", force_download=True)
```

The model will look for the preferred device to run the model on (per default it will use the GPU if available, otherwise it will use the CPU). But you can also specify the device manually:

```python
calculator = ShiftML("ShiftML4", device="cpu")  # run always on CPU

calculator = ShiftML("ShiftML4", device="cuda")  # run always on GPU
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
### ShiftML3 and ShiftML4 – Frequently Asked Questions


<details>
<summary><strong>ShiftML3/ShiftML4 predictions aren’t identical for magnetically equivalent atoms. Why?</strong></summary>

ShiftML3/ShiftML4 is built on the **Point Edge Transformer (PET)** model, which is *not perfectly rotationally invariant*.
This can introduce tiny, random differences for atoms that are magnetically equivalent.
We have verified that these fluctuations are minor and do **not** harm overall accuracy.

> **Tip – get identical shielding predictions**
> Average the predictions over all magnetically equivalent atoms.

</details>

---

<details>
<summary><strong>ShiftML3/ShiftML4 shows large errors versus my GIPAW-DFT shieldings. What’s going on?</strong></summary>

Chemical-shielding calculations are *very* sensitive to the **code and convergence parameters** used.
Only compare ShiftML3/ShiftML4 to GIPAW-DFT data generated with *exactly* the same settings as the training set.

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

Matthias Kellner
Yuxuan Zhang
Ruben Rodriguez Madrid
Guillaume Fraux

This project is [maintained](https://github.com/lab-cosmo/.github/blob/main/Maintainers.md) by [@bananenpampe](https://github.com/bananenpampe), who will reply to issues and pull requests opened on this repository as soon as possible. You can mention them directly if you did not receive an answer after a couple of days.

## References

This package is based on the following papers:

- Chemical shifts in molecular solids by machine learning - Paruzzo et al. [[1](https://doi.org/10.1038%2Fs41467-018-06972-x)]
- A Bayesian approach to NMR crystal structure determination - Engel et al. [[2](https://doi.org/10.1039%2Fc9cp04489b)]
- A Machine Learning Model of Chemical Shifts for Chemically and\
Structurally Diverse Molecular Solids - Cordova et al. [[3](https://doi.org/10.1021/acs.jpcc.2c03854)]
- A deep learning model for chemical shieldings in molecular organic solids including anisotropy - Kellner, Holmes, Rodriguez Madrid, Viscosi, Zhang, Emsley, Ceriotti  [[4](https://doi.org/10.1021/acs.jpclett.5c01819)]
