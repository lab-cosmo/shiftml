import logging
import os
import urllib.request
import numpy as np

from metatensor.torch.atomistic import ModelOutput
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from platformdirs import user_cache_path

from shiftml.utils.tensorial import T_sym_np_inv, symmetrize

# For now we set the logging level to DEBUG
logformat = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logformat)


url_resolve = {
    "ShiftML1.1rev": "https://tinyurl.com/msnss4ds",
    "ShiftML2.0rev": "https://tinyurl.com/3axupmsd",
    "ShiftML2.1dev": "https://zenodo.org/record/14920547/files/model.pt?download=1",
    "ShiftML2.1dev_csa": "https://zenodo.org/records/14962123/files/model_csa.pt?download=1",
}

cs_iso_output = {"mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True)}

cs_iso_ensemble_output = {
        "mtt::cs_iso": ModelOutput(quantity="", unit="ppm", per_atom=True),
        "mtt::cs_iso_std": ModelOutput(quantity="", unit="ppm", per_atom=True),
        "mtt::cs_iso_ensemble": ModelOutput(quantity="", unit="ppm", per_atom=True),
    },

resolve_outputs = {
    "ShiftML1.1rev": cs_iso_output,
    "ShiftML2.1dev": cs_iso_output,
    "ShiftML2.0rev": cs_iso_ensemble_output,
    "ShiftML2.1dev_ensemble": cs_iso_output,
    "ShiftML2.1csa_dev_ensemble": cs_iso_output,
    "ShiftML2.1dev_csa": cs_iso_output,
}

resolve_fitted_species = {
    "ShiftML1.1rev": set([1, 6, 7, 8, 16]),
    "ShiftML2.0rev": set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]),
    "ShiftML2.1dev": set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]),
    "ShiftML2.1dev_ensemble": set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]),
    "ShiftML2.1csa_dev_ensemble": set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]),
    "ShiftML2.1dev_csa": set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20]),
}

requires_metatrain = ["ShiftML2.1dev_ensemble", "ShiftML2.1dev_csa"]

# prepares ensemble model
for i in range(1,8):
    url_resolve["ShiftML2.1dev" + str(i)] = f"https://zenodo.org/records/14920832/files/model_{i}.pt?download=1"
    resolve_fitted_species["ShiftML2.1dev" + str(i)] = set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20])
    resolve_outputs["ShiftML2.1dev" + str(i) ] = cs_iso_output
    requires_metatrain.append("ShiftML2.1dev" + str(i))

# prepares cs_ensemble model
for i in range(0,8):
    url_resolve["ShiftML2.1csa_dev" + str(i)] = f"https://zenodo.org/records/15079415/files/model_{i}.pt?download=1"
    resolve_fitted_species["ShiftML2.1csa_dev" + str(i)] = set([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20])
    resolve_outputs["ShiftML2.1csa_dev" + str(i)] = cs_iso_output
    requires_metatrain.append("ShiftML2.1csa_dev" + str(i))

def is_fitted_on(atoms, fitted_species):
    if not set(atoms.get_atomic_numbers()).issubset(fitted_species):
        raise ValueError(
            f"Model is fitted only for the following atomic numbers:\
            {fitted_species}. The atomic numbers in the atoms object are:\
            {set(atoms.get_atomic_numbers())}. Please provide an atoms object\
            with only the fitted species."
        )

def ShiftML(model_version, force_download=False):
        """
        Initialize the ShiftML calculator

        Parameters
        ----------
        model_version : str
            The version of the ShiftML model to use. Supported versions are
            "ShiftML1.1rev" and "ShiftML2.0rev" "ShiftML2.1devel".
        force_download : bool, optional
            If True, the model will be downloaded even if it is already in the cache.
            The chache-dir will be determined via the platformdirs library and should
            comply with user settings such as XDG_CACHE_HOME.
            Default is False.
        """

        # its not perfect, it is what it is...
        if model_version in ["ShiftML2.1dev_ensemble", "ShiftML2.1csa_dev_ensemble"]:
            model_version = model_version.replace("_ensemble", "")
            model_list = []
            
            if model_version == "ShiftML2.1csa_dev":
                for i in range(0,8):
                    model_list.append(ShiftML_model(model_version + str(i), force_download=force_download))
            elif model_version == "ShiftML2.1dev":
                for i in range(1,8):
                    model_list.append(ShiftML_model(model_version + str(i), force_download=force_download))
            
            return ShiftML_ensemble(model_list)
        
        else:
            return  ShiftML_model(model_version, force_download=force_download)



class ShiftML_ensemble:
    def __init__(self, model_list):
        """
        Initializes an ensemble of ShiftML models
        """
        self.models = model_list

    def get_cs_iso(self, atoms):
        """
        Compute the shielding values for the given atoms object
        """

        cs_isos = []

        for model in self.models:
            out = model.get_cs_iso(atoms)
            cs_isos.append(out)

        cs_iso = np.mean(np.hstack(cs_isos),axis=1)

        return cs_iso

    def get_cs_iso_ensemble(self, atoms):
        
        cs_isos = []

        for model in self.models:
            out = model.get_cs_iso(atoms)
            cs_isos.append(out)

        cs_iso = np.hstack(cs_isos)

        return cs_iso
    
    def get_cs_tensor_ensemble(self, atoms, return_symmetric=True):
        cs_tensors = []
        
        for model in self.models:
            out = model.get_cs_tensor(atoms, return_symmetric=return_symmetric)
            cs_tensors.append(out)
        
        cs_tensors = np.stack(cs_tensors,axis=-1)

        return cs_tensors
    
class ShiftML_model(MetatensorCalculator):
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
            "ShiftML1.1rev" and "ShiftML2.0rev" "ShiftML2.1devel".
        force_download : bool, optional
            If True, the model will be downloaded even if it is already in the cache.
            The chache-dir will be determined via the platformdirs library and should
            comply with user settings such as XDG_CACHE_HOME.
            Default is False.
        """

        if model_version in ["ShiftML1.1rev", "ShiftML2.0rev"]:
            try:
                # The rascline import is necessary because
                # it is required for the scripted model
                import featomic.torch

                logging.info("featomic version: {}".format(featomic.torch.__version__))
                logging.info("featomic-torch is installed, importing featomic-torch")

                assert (
                    featomic.torch.__version__ == "0.1.0.dev0" #"0.1.0.dev597"
                ), "wrong featomic-torch installed"

            except ImportError:
                raise ImportError(
                    "featomic-torch is required for featomic based ShiftML calculators,\
                    please install it using\
                    pip install git+https://github.com/metatensor/featomic#subdirectory\
                    =python/featomic-torch"
                )
        ### lol why doesnt this break?????

        elif model_version in ["ShiftML2.1dev","ShiftML2.1dev_csa"]:
            try: # 0.1.dev300+g7a465bf
                import metatrain
                logging.info("metatrain version: {}".format(metatrain.__version__))
                logging.info("metatrain is installed, importing metatrain")

                assert (
                    metatrain.__version__ == "0.1.dev300+g7a465bf"
                ), "wrong featomic-torch installed"
            
            except ImportError:
                raise ImportError(
                    "metatrain is required for nanoPET based ShiftML calculators,\
                    please install it using\
                    pip install metatrain[nanopet]@git+https://github.com/metatensor/\
                    metatrain@7a465bf"
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
        cs_iso = out["mtt::cs_iso"].block(0).values.detach().to("cpu").numpy()

        return cs_iso

    def get_cs_iso_std(self, atoms):
        assert (
            "mtt::cs_iso_std" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        is_fitted_on(atoms, self.fitted_species)

        out = self.run_model(atoms, self.outputs)
        cs_iso_std = out["mtt::cs_iso_std"].block(0).values.detach().to("cpu").numpy()

        return cs_iso_std

    def get_cs_iso_ensemble(self, atoms):
        assert (
            "mtt::cs_iso_ensemble" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        is_fitted_on(atoms, self.fitted_species)

        out = self.run_model(atoms, self.outputs)
        cs_iso_ensemble = out["mtt::cs_iso_ensemble"].block(0).values.detach().to("cpu").numpy()

        return cs_iso_ensemble
    
    def get_cs_tensor(self, atoms, return_symmetric=True):
        assert (
            "mtt::cs_iso" in self.outputs.keys()
        ), "model does not support chemical shielding prediction"

        is_fitted_on(atoms, self.fitted_species)

        out = self.run_model(atoms, self.outputs)
        out =  out["mtt::cs_iso"].components_to_properties(["o3_mu"])

        pred_vals = np.concatenate([block.values.to("cpu").numpy() for block in out.blocks()],axis=1) @ T_sym_np_inv.T

        if return_symmetric:
            pred_vals = symmetrize(pred_vals)

        return pred_vals 
