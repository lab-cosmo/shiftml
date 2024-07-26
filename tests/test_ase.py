# TODO: test for rotational invariance, translation invariance,
# and permutation, as well as size extensivity
import numpy as np
import pytest
from ase.build import bulk

from shiftml.ase import ShiftML

expected_output = np.array([137.5415, 137.5415])


def test_shiftml1_regression():
    """Regression test for the ShiftML1.0 model."""

    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML("ShiftML1.0", force_download=True)
    out = model.get_cs_iso(frame)

    assert np.allclose(
        out.flatten(), expected_output
    ), "ShiftML1 failed regression test"


def test_shiftml1_rotational_invariance():
    """Rotational invariance test for the ShiftML1.0 model."""

    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML("ShiftML1.0")
    out = model.get_cs_iso(frame)

    assert np.allclose(
        out.flatten(), expected_output
    ), "ShiftML1 failed regression test"

    # Rotate the frame by 90 degrees about the z-axis
    frame.rotate(90, "z")

    out_rotated = model.get_cs_iso(frame)

    assert np.allclose(
        out_rotated.flatten(), expected_output
    ), "ShiftML1 failed rotational invariance test"


def test_shiftml1_size_extensivity_test():
    """Test ShiftML1.0 for translational invariance."""

    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML("ShiftML1.0")
    out = model.get_cs_iso(frame)

    assert np.allclose(
        out.flatten(), expected_output
    ), "ShiftML1 failed regression test"

    frame = frame * (2, 1, 1)
    out = model.get_cs_iso(frame)

    assert np.allclose(
        out.flatten(), np.stack([expected_output, expected_output]).flatten()
    ), "ShiftML1 failed size extensivity test"


def test_shftml1_fail_invalid_species():
    """Test ShiftML1.o for non-fitted species"""

    frame = bulk("Si", "diamond", a=3.566)
    model = ShiftML("ShiftML1.0")
    with pytest.raises(ValueError) as exc_info:
        model.get_cs_iso(frame)

    assert exc_info.type == ValueError
    assert "Model is fitted only for the following atomic numbers:" in str(
        exc_info.value
    )
