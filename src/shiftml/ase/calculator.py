import os
import urllib.request

from metatensor.torch.atomistic import ModelOutput
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator

url_resolve = {
    "ShiftML1.0": "https://tinyurl.com/3xwec68f",
}

resolve_outputs = {
    "ShiftML1.0": {"mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True)},
}

resolve_fitted_species = {
    "ShiftML1.0": set([1, 6, 7, 8, 16]),
}


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
            Default is False.
        """

        try:
            # The rascline import is necessary because
            # it is required for the scripted model
            import rascaline.torch

            print(rascaline.torch.__version__)
            print("rascaline-torch is installed, importing rascaline-torch")

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
            print("Found model version in url_resolve")
            print("Resolving model version to model files at url: ", url)
        except KeyError:
            raise ValueError(
                f"Model version {model_version} is not supported.\
                    Supported versions are {list(url_resolve.keys())}"
            )

        cachedir = os.path.expanduser(
            os.path.join("~", ".cache", "shiftml", str(model_version))
        )

        # check if model is already downloaded
        try:
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)
            model_file = os.path.join(cachedir, "model.pt")

            if os.path.exists(model_file) and force_download:
                print(
                    f"Found {model_version} in cache, but force_download is set to True"
                )
                print(f"Removing {model_version} from cache and downloading it again")
                os.remove(os.path.join(cachedir, "model.pt"))
                download = True

            else:
                if os.path.exists(model_file):
                    print(
                        f"Found {model_version}  in cache,\
                         and importing it from here: {cachedir}"
                    )
                    download = False
                else:
                    print("Model not found in cache, downloading it")
                    download = True

            if download:
                urllib.request.urlretrieve(url, os.path.join(cachedir, "model.pt"))
                print(f"Downloaded {model_version} and saved to {cachedir}")

        except urllib.error.URLError as e:
            print(
                f"Failed to download {model_version} from {url}. URL Error: {e.reason}"
            )
            raise e
        except urllib.error.HTTPError as e:
            print(
                f"Failed to download {model_version} from {url}.\
                  HTTP Error: {e.code} - {e.reason}"
            )
            raise e
        except Exception as e:
            print(
                f"An unexpected error occurred while downloading\
                  {model_version} from {url}: {e}"
            )
            raise e

        super().__init__(model_file)

    def get_cs_iso(self, atoms):
        """
        Compute the shielding values for the given atoms object
        """

        assert (
            "mtt::cs_iso" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        if not set(atoms.get_atomic_numbers()).issubset(self.fitted_species):
            raise ValueError(
                f"Model is fitted only for the following atomic numbers:\
                {self.fitted_species}. The atomic numbers in the atoms object are:\
                {set(atoms.get_atomic_numbers())}. Please provide an atoms object\
                with only the fitted species."
            )

        out = self.run_model(atoms, self.outputs)
        cs_iso = out["mtt::cs_iso"].block(0).values.detach().numpy()

        return cs_iso
