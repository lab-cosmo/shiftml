import numpy as np
from ase.build import bulk

from shiftml.ase import ShiftML

expected_outputs = {
    "ShiftML1.0": np.array([137.5415, 137.5415]),
    "ShiftML1.1": np.array([163.07251, 163.07251]),
}


def test_shiftml1_regression():
    """Regression test for the ShiftML1.0 model."""

    MODEL = "ShiftML1.0"

    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(MODEL, force_download=True)
    out = model.get_cs_iso(frame)

    assert np.allclose(
        out.flatten(), expected_outputs[MODEL]
    ), "ShiftML1.0 failed regression test"


def test_shiftml1_1_regression():
    """Regression test for the ShiftML1.1 model."""

    MODEL = "ShiftML1.1"

    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(MODEL, force_download=True)
    out = model.get_cs_iso(frame)

    assert np.allclose(
        out.flatten(), expected_outputs[MODEL]
    ), "ShiftML1.1 failed regression test"
