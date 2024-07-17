import os

os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"

import shiftml  # noqa: E402

project = "ShiftML"
copyright = "2024, ShiftML developers"
author = "TODO"
release = shiftml.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "rascaline": ("https://luthaf.fr/rascaline/latest/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}
