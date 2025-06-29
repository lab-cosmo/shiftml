# TODO: test for rotational invariance, translation invariance,
# and permutation, as well as size extensivity
import numpy as np
import pytest
from ase.build import bulk

from shiftml.ase import ShiftML

expected_outputs = {"ShiftML3": np.array([70.39079043, 70.4060931])}

expected_outputs_tensors = {
    "ShiftML3": np.array(
        [
            [
                [70.78904458, -2.59054161, -0.50549243],
                [-2.59054161, 67.21772471, -2.20606105],
                [-0.50549243, -2.20606105, 73.16562596],
            ],
            [
                [70.66452058, -1.21551709, -1.11634899],
                [-1.21551709, 67.31813798, -1.05462152],
                [-1.11634899, -1.05462152, 73.23578343],
            ],
        ]
    ),
    "ShiftML30": np.array(
        [
            [
                [55.65042743, -1.63128062, -0.32191175],
                [-1.63128062, 64.67196237, -9.39962522],
                [-0.32191175, -9.39962522, 60.94139457],
            ],
            [
                [55.24493472, 2.82685016, -1.38101845],
                [2.82685016, 64.60677816, -4.46976634],
                [-1.38101845, -4.46976634, 60.54103779],
            ],
        ]
    ),
}


expected_outputs_cs_iso_ensemble = {
    "ShiftML3": np.array(
        [
            [
                60.42126146,
                57.24088394,
                119.28280679,
                5.62315253,
                116.85640454,
                97.67294283,
                78.86602387,
                27.1628475,
            ],
            [
                60.13091689,
                57.4300539,
                119.20077118,
                6.10752558,
                117.19963794,
                97.69625321,
                78.82573726,
                26.65784881,
            ],
        ]
    )
}


def test_diamond_regression():
    """Regression test for ShiftML models."""

    frame = bulk("C", "diamond", a=3.566)

    for key, value in expected_outputs.items():

        model = ShiftML(key, force_download=True, device="cpu")
        out = model.get_cs_iso(frame)

        assert np.allclose(out.flatten(), value), f"{key} failed regression test"


def test_shiftml1_size_extensivity_test():
    """Test ShiftML mdodel for size extensivity"""

    frame = bulk("C", "diamond", a=3.566)
    frame = frame.repeat((2, 1, 1))

    for key, value in expected_outputs.items():

        model = ShiftML(key, device="cpu")
        out = model.get_cs_iso(frame)

        assert np.allclose(
            out.flatten(), np.stack([value, value]).flatten()
        ), f"{key} failed regression test"


def test_shiftml3_tensors():
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML("ShiftML3", device="cpu")
    cs_tensor = model.get_cs_tensor(frame, return_symmetric=True)
    assert cs_tensor.shape == (2, 3, 3), "CS tensor shape mismatch"

    # assert that the tensor is symmetric
    assert np.allclose(
        cs_tensor, cs_tensor.transpose(0, 2, 1)
    ), "CS tensor is not symmetric"

    assert np.allclose(
        cs_tensor, expected_outputs_tensors["ShiftML3"], rtol=1e-4
    ), "CS tensor values do not match expected output"


def test_shiftml3_single_model_tensors():
    """Regression test of one of the ShiftML3 models (model 0)"""
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML("ShiftML30", device="cpu")
    cs_tensor = model.get_cs_tensor(frame, return_symmetric=True).reshape((2,3,3))
    assert cs_tensor.shape == (2, 3, 3), "CS tensor shape mismatch"

    # assert that the tensor is symmetric
    assert np.allclose(
        cs_tensor, cs_tensor.transpose(0, 2, 1)
    ), "CS tensor is not symmetric"

    assert np.allclose(
        cs_tensor, expected_outputs_tensors["ShiftML30"], rtol=1e-3
    ), "CS tensor values do not match expected output"


def test_shftml3_fail_invalid_species():
    """Test ShiftML1.1rev for non-fitted species"""

    frame = bulk("Si", "diamond", a=3.566)
    model = ShiftML("ShiftML3", device="cpu")
    with pytest.raises(ValueError) as exc_info:
        model.get_cs_iso(frame)

    assert exc_info.type == ValueError
    assert "Model is fitted only for the following atomic numbers:" in str(
        exc_info.value
    )
