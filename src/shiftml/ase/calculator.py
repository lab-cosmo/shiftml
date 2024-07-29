import logging
import os
import urllib.request

from metatensor.torch.atomistic import ModelOutput
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from platformdirs import user_cache_path

# For now we set the logging level to DEBUG
logformat = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logformat)


url_resolve = {
    "ShiftML1.0": "https://tinyurl.com/3xwec68f",
    "ShiftML1.1": "https://tinyurl.com/53ymkhvd",
    "ShiftML2.0": "https://tinyurl.com/bdcp647w",
}

resolve_outputs = {
    "ShiftML1.0": {"mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True)},
    "ShiftML1.1": {"mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True)},
    "ShiftML2.0": {
        "mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True),
        "mtt::cs_iso_std": ModelOutput(quantity="", unit="ppm", per_atom=True),
        "mtt::cs_iso_ensemble": ModelOutput(quantity="", unit="ppm", per_atom=True),
    },
}

resolve_fitted_species = {
    "ShiftML1.0": set([1, 6, 7, 8, 16]),
    "ShiftML1.1": set([1, 6, 7, 8, 16]),
    "ShiftML2.0": set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]),
}


def is_fitted_on(atoms, fitted_species):
    if not set(atoms.get_atomic_numbers()).issubset(fitted_species):
        raise ValueError(
            f"Model is fitted only for the following atomic numbers:\
            {fitted_species}. The atomic numbers in the atoms object are:\
            {set(atoms.get_atomic_numbers())}. Please provide an atoms object\
            with only the fitted species."
        )


class ShiftML(MetatensorCalculator):
    """
    ShiftML calculator for ASE
    """

    def __init__(self, model_version, force_download=False):
        """
        Initialize the ShiftML calculator

        Parameters
        ----------
        model_version : str
            The version of the ShiftML model to use. Supported versions are
            "ShiftML1.0".
        force_download : bool, optional
            If True, the model will be downloaded even if it is already in the cache.
            The chache-dir will be determined via the platformdirs library and should
            comply with user settings such as XDG_CACHE_HOME.
            Default is False.
        """

        try:
            # The rascline import is necessary because
            # it is required for the scripted model
            import rascaline.torch

            logging.info("rascaline version: {}".format(rascaline.torch.__version__))
            logging.info("rascaline-torch is installed, importing rascaline-torch")

            assert (
                rascaline.torch.__version__ == "0.1.0.dev558"
            ), "wrong rascaline-torch installed"

        except ImportError:
            raise ImportError(
                "rascaline-torch is required for ShiftML calculators,\
                 please install it using\
                 pip install git+https://github.com/luthaf/rascaline#subdirectory\
                 =python/rascaline-torch"
            )

        try:
            url = url_resolve[model_version]
            self.outputs = resolve_outputs[model_version]
            self.fitted_species = resolve_fitted_species[model_version]
            logging.info("Found model version in url_resolve")
            logging.info(
                "Resolving model version to model files at url: {}".format(url)
            )
        except KeyError:
            raise ValueError(
                f"Model version {model_version} is not supported.\
                    Supported versions are {list(url_resolve.keys())}"
            )

        cachedir = os.path.expanduser(
            os.path.join(user_cache_path(), "shiftml", str(model_version))
        )

        # check if model is already downloaded
        try:
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)
            model_file = os.path.join(cachedir, model_version + ".pt")

            if os.path.exists(model_file) and force_download:
                logging.info(
                    "Found {} in cache, but force_download is set to True".format(
                        model_version
                    )
                )
                logging.info(
                    "Removing {} from cache and downloading it again".format(
                        model_version
                    )
                )
                os.remove(model_file)
                download = True

            else:
                if os.path.exists(model_file):
                    logging.info(
                        "Found {}  in cache,\
                         and importing it from here: {}".format(
                            model_version, cachedir
                        )
                    )
                    download = False
                else:
                    logging.info("Model not found in cache, downloading it")
                    download = True

            if download:
                urllib.request.urlretrieve(url, model_file)
                logging.info(
                    "Downloaded {} and saved to {}".format(model_version, cachedir)
                )

        except urllib.error.URLError as e:
            logging.error(
                "Failed to download {} from {}. URL Error: {}".format(
                    model_version, url, e.reason
                )
            )
            raise e
        except urllib.error.HTTPError as e:
            logging.error(
                "Failed to download {} from {}.\
                  HTTP Error: {} - {}".format(
                    model_version, url, e.code, e.reason
                )
            )
            raise e
        except Exception as e:
            logging.error(
                "An unexpected error occurred while downloading\
                  {} from {}: {}".format(
                    model_version, url, e
                )
            )
            raise e

        super().__init__(model_file)
        self.model_version = model_version

    def get_cs_iso(self, atoms):
        """
        Compute the shielding values for the given atoms object
        """
        assert (
            "mtt::cs_iso" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        is_fitted_on(atoms, self.fitted_species)

        out = self.run_model(atoms, self.outputs)
        cs_iso = out["mtt::cs_iso"].block(0).values.detach().numpy()

        return cs_iso

    def get_cs_iso_std(self, atoms):
        assert (
            "mtt::cs_iso_std" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        is_fitted_on(atoms, self.fitted_species)

        out = self.run_model(atoms, self.outputs)
        cs_iso_std = out["mtt::cs_iso_std"].block(0).values.detach().numpy()

        return cs_iso_std

    def get_cs_iso_ensemble(self, atoms):
        assert (
            "mtt::cs_iso_ensemble" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        is_fitted_on(atoms, self.fitted_species)

        out = self.run_model(atoms, self.outputs)
        cs_iso_ensemble = out["mtt::cs_iso_ensemble"].block(0).values.detach().numpy()

        return cs_iso_ensemble
