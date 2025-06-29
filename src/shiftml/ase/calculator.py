import logging
import os
import urllib.request

import numpy as np
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator
from platformdirs import user_cache_path

from shiftml.utils.tensorial import T_sym_np_inv, symmetrize

# For now we set the logging level to INFO
logformat = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=logformat)


url_resolve = {}

cs_iso_output = {"mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True)}

resolve_outputs = {
    "ShiftML3": cs_iso_output,
}

resolve_fitted_species = {
    "ShiftML3": set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]),
}

# prepares cs_ensemble model
for i in range(0, 8):
    url_resolve["ShiftML3" + str(i)] = (
        f"https://zenodo.org/records/15767390/files/model_{i}.pt?download=1"
    )
    resolve_fitted_species["ShiftML3" + str(i)] = set(
        [1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]
    )
    resolve_outputs["ShiftML3" + str(i)] = cs_iso_output


def is_fitted_on(atoms, fitted_species):
    if not set(atoms.get_atomic_numbers()).issubset(fitted_species):
        raise ValueError(
            f"Model is fitted only for the following atomic numbers:\
            {fitted_species}. The atomic numbers in the atoms object are:\
            {set(atoms.get_atomic_numbers())}. Please provide an atoms object\
            with only the fitted species."
        )


def ShiftML(model_version, force_download=False, device=None):
    """
    Initialize the ShiftML calculator

    Parameters
    ----------
    model_version : str
        The version of the ShiftML model to use. Supported versions are
        "ShiftML3"
    force_download : bool, optional
        If True, the model will be downloaded even if it is already in the cache.
        The chache-dir will be determined via the platformdirs library and should
        comply with user settings such as XDG_CACHE_HOME.
        Default is False.
    device : str, optional
        The device to use for the model. If None, the preferred device will be saught
        from the environment (e.g. CUDA if available), it will fall back to CPU if no
        preffered device is found.
        If you want to use a specific device, you can set it to "cpu" or "cuda".
        Default is None.
    """

    # its not perfect, it is what it is...
    if model_version in ["ShiftML3"]:
        model_list = []
        for i in range(0, 8):
            model_list.append(
                ShiftML_model(
                    model_version + str(i), force_download=force_download, device=device
                )
            )

        return ShiftML_ensemble(model_list)

    else:
        return ShiftML_model(
            model_version, force_download=force_download, device=device
        )


class ShiftML_ensemble:
    def __init__(self, model_list):
        """
        Initializes an ensemble of ShiftML models
        """
        self.models = model_list

    def get_cs_tensor_ensemble(self, atoms, return_symmetric=True):
        cs_tensors = []

        for model in self.models:
            out = model.get_cs_tensor(atoms, return_symmetric=return_symmetric)
            cs_tensors.append(out)

        cs_tensors = np.stack(cs_tensors, axis=-1)

        # last dimension is ensemble member shape
        cs_tensors = cs_tensors.reshape(-1, 3, 3, cs_tensors.shape[-1])

        return cs_tensors

    def get_cs_iso_ensemble(self, atoms):

        cs_tensors = self.get_cs_tensor_ensemble(atoms, return_symmetric=True)
        cs_isos = np.trace(cs_tensors, axis1=1, axis2=2) / 3

        return cs_isos

    def get_cs_iso(self, atoms):
        """
        Compute the shielding values for the given atoms object
        """

        cs_isos = self.get_cs_iso_ensemble(atoms)

        cs_iso = np.mean(cs_isos, axis=-1)

        return cs_iso

    def get_cs_tensor(self, atoms, return_symmetric=True):
        """
        Compute the shielding tensors for the given atoms object
        """

        cs_tensors = self.get_cs_tensor_ensemble(
            atoms, return_symmetric=return_symmetric
        )

        cs_tensors = np.mean(cs_tensors, axis=-1).reshape(-1, 3, 3)

        return cs_tensors


class ShiftML_model(MetatomicCalculator):
    """
    ShiftML calculator for ASE
    """

    def __init__(self, model_version, force_download=False, device=None):
        """
        Initialize the ShiftML calculator

        Parameters
        ----------
        model_version : str
            The version of the ShiftML model to use.
        force_download : bool, optional
            If True, the model will be downloaded even if it is already in the cache.
            The chache-dir will be determined via the platformdirs library and should
            comply with user settings such as XDG_CACHE_HOME.
            Default is False.
        device : str, optional
            The device to use for the model. If None, the preferred
            device will be sought from the environment (e.g. CUDA if available),
            it will fall back to CPU if no preferred device is found.
            If you want to use a specific device, you can set it to "cpu" or "cuda".
            Default is None.
        """

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

        super().__init__(
            model_file,
            device=device,
        )

        self.model_version = model_version

    def get_cs_tensor(self, atoms, return_symmetric=True):
        assert (
            "mtt::cs_iso" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        is_fitted_on(atoms, self.fitted_species)

        out = self.run_model(atoms, self.outputs)

        # TODO: currently ShiftML3 predicts tensors with the label  "mtt::cs_iso"
        # later this should be changed to "mtt::cs_tensor"
        out = out["mtt::cs_iso"].components_to_properties(["o3_mu"])

        pred_vals = (
            np.concatenate(
                [block.values.to("cpu").numpy() for block in out.blocks()], axis=1
            )
            @ T_sym_np_inv.T
        )

        if return_symmetric:
            pred_vals = symmetrize(pred_vals)

        return pred_vals
