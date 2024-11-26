import numpy as np
from ase.build import bulk

from shiftml.ase import ShiftML

expected_outputs = {
    "ShiftML1.1rev": np.array([137.5415, 137.5415]),
}


def test_shiftml1_1_regression():
    """Regression test for the ShiftML1.1 model."""

    MODEL = "ShiftML1.1rev"

    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(MODEL, force_download=True)
    out = model.get_cs_iso(frame)

    assert np.allclose(
        out.flatten(), expected_outputs[MODEL]
    ), "ShiftML1.1 failed regression test"
