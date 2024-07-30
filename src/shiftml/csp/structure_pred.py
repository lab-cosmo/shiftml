import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from shiftml.ase import ShiftML


def load_experimental(filename, skiprows=1):
    """
    Function to import experimental .csv file and
    convert to atomic indexes and chemical shieldings

    Parameters
    ----------
    filname : str
        name of the .csv file with path
    skiprows : int
        number of the rows to skip, default 1

    Outputs
    -------
    list_atom : np.array(List[str])
        list of the atomic indexes, in string array
    list_cs: np.array(List[str])
        chemical sheilding for each corresponding index, in float array
    """
    exp_data = pd.read_csv(filename, skiprows=skiprows)
    atom_label = exp_data["Atom Label"]
    atom_cs = exp_data["^(1)Hdelta(ppm)"]
    list_atom = atom_label.to_list()
    list_cs = atom_cs.array
    return list_atom, list_cs


def extract_from_array(array, num_atoms, list_atom):
    """
    Function to extract regression array from a given
    symbol list of atoms and data exported from the model
    """
    num_molecules = int(len(array) / num_atoms)
    data_fit = array.reshape((-1, num_molecules))[:, 0]
    X = []
    for atom_string in list_atom:
        label_list = atom_string.split(",")
        if len(label_list) == 1:
            label = int(label_list[0])
            X.append(data_fit[label - 1])
        else:
            X.append(
                sum([data_fit[int(label_str) - 1] for label_str in label_list])
                / len(label_list)
            )
    return np.array(X)


def structure_prediction(
    model_version, frames, list_atom, list_cs, GIPAW_avail=True, cs_sym="CS"
):
    """
    Function to select the suitable structures
    based on a set of candidate structures,
    given rmse of the linear regression results

    Parameters
    ----------
    model_version : str
        The version of the ShiftML model to use. Supported versions are
        "ShiftML1.0", "ShiftML1.1", and "ShiftML2.0".
    frames : List[ase.Atoms]
        A list of candidate structures.
    list_atom : np.array(List[str])
        An array of atom symbols included in the structure.
    list_cs: np.array(List[float])
        An array of chemical shielding values corresponding to the atom symbols.
    """
    calculator = ShiftML(model_version)
    number_list = list_atom[-1].split(",")
    num_atoms = float(number_list[-1])
    rmse_rec1 = np.array([])
    rmse_rec2 = np.array([])

    for frame in frames:
        Y = list_cs
        atom_label = frame.get_atomic_numbers() == 1
        array = calculator.get_cs_iso(frame).ravel()[atom_label]
        X = extract_from_array(array, num_atoms, list_atom)
        slope = -1
        intercept = np.mean(Y) - slope * np.mean(X)
        rmse = root_mean_squared_error(slope * X + intercept, Y)
        rmse_rec1 = np.append(rmse_rec1, rmse)
        if GIPAW_avail:
            array = frame[atom_label].arrays[cs_sym].ravel()
            X = extract_from_array(array, num_atoms, list_atom)
            slope = -1
            intercept = np.mean(Y) - slope * np.mean(X)
            rmse = root_mean_squared_error(slope * X + intercept, Y)
            rmse_rec2 = np.append(rmse_rec2, rmse)
            rmse_rec = (rmse_rec1, rmse_rec2)
        else:
            rmse_rec = rmse_rec1

    return rmse_rec
