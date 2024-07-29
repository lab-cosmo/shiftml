# TODO: test for rotational invariance, translation invariance,
# and permutation, as well as size extensivity
import numpy as np
import pytest
from ase.build import bulk

from shiftml.ase import ShiftML

expected_output = np.array([137.5415, 137.5415])
expected_ensemble_v2 = np.array(
    [
        [
            114.8194808959961,
            113.47244262695312,
            117.47064208984375,
            115.61190795898438,
            130.88909912109375,
            131.42332458496094,
            120.96844482421875,
            115.95867919921875,
            116.98180389404297,
            135.3658447265625,
            120.45016479492188,
            123.00967407226562,
            137.23724365234375,
            129.23104858398438,
            131.00619506835938,
            130.82601928710938,
            121.90162658691406,
            120.66400909423828,
            109.59469604492188,
            118.66798400878906,
            126.18386840820312,
            124.9156494140625,
            120.90362548828125,
            106.26658630371094,
            128.32107543945312,
            125.82593536376953,
            121.3394775390625,
            127.37902069091797,
            122.92572784423828,
            126.26400756835938,
            112.87037658691406,
            112.48919677734375,
            126.00082397460938,
            109.98661804199219,
            110.7204818725586,
            107.30191040039062,
            113.85182189941406,
            110.24645233154297,
            133.27935791015625,
            126.40534973144531,
            133.42047119140625,
            112.2728271484375,
            126.27506256103516,
            117.58969116210938,
            119.17208099365234,
            121.65959167480469,
            115.62092590332031,
            118.12762451171875,
            119.478271484375,
            137.32974243164062,
            120.26103210449219,
            118.25013732910156,
            121.78120422363281,
            125.66693115234375,
            112.0889892578125,
            115.92691802978516,
            121.31621551513672,
            118.76759338378906,
            126.86924743652344,
            129.01571655273438,
            109.53144073486328,
            110.71353149414062,
            125.9607925415039,
            108.36444091796875,
        ],
        [
            114.8194808959961,
            113.47244262695312,
            117.47064208984375,
            115.61190795898438,
            130.88909912109375,
            131.42332458496094,
            120.96844482421875,
            115.95867919921875,
            116.98180389404297,
            135.3658447265625,
            120.45016479492188,
            123.00967407226562,
            137.23724365234375,
            129.23104858398438,
            131.00619506835938,
            130.82601928710938,
            121.90162658691406,
            120.66400909423828,
            109.59469604492188,
            118.66798400878906,
            126.18386840820312,
            124.9156494140625,
            120.90362548828125,
            106.26658630371094,
            128.32107543945312,
            125.82593536376953,
            121.3394775390625,
            127.37902069091797,
            122.92572784423828,
            126.26400756835938,
            112.87037658691406,
            112.48919677734375,
            126.00082397460938,
            109.98661804199219,
            110.7204818725586,
            107.30191040039062,
            113.85182189941406,
            110.24645233154297,
            133.27935791015625,
            126.40534973144531,
            133.42047119140625,
            112.2728271484375,
            126.27506256103516,
            117.58969116210938,
            119.17208099365234,
            121.65959167480469,
            115.62092590332031,
            118.12762451171875,
            119.478271484375,
            137.32974243164062,
            120.26103210449219,
            118.25013732910156,
            121.78120422363281,
            125.66693115234375,
            112.0889892578125,
            115.92691802978516,
            121.31621551513672,
            118.76759338378906,
            126.86924743652344,
            129.01571655273438,
            109.53144073486328,
            110.71353149414062,
            125.9607925415039,
            108.36444091796875,
        ],
    ]
)
expected_mean_v2 = np.array([120.85137, 120.85137])
expected_std_v2 = np.array([7.7993703, 7.7993703])


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
    """Test ShiftML1.0 for non-fitted species"""

    frame = bulk("Si", "diamond", a=3.566)
    model = ShiftML("ShiftML1.0")
    with pytest.raises(ValueError) as exc_info:
        model.get_cs_iso(frame)

    assert exc_info.type == ValueError
    assert "Model is fitted only for the following atomic numbers:" in str(
        exc_info.value
    )


def test_shiftml2_regression_mean():
    """Regression test for the ShiftML2.0 model."""

    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML("ShiftML2.0", force_download=True)
    out_mean = model.get_cs_iso(frame)
    out_std = model.get_cs_iso_std(frame)
    out_ensemble = model.get_cs_iso_ensemble(frame)

    assert np.allclose(
        out_mean.flatten(), expected_mean_v2
    ), "ShiftML2 failed regression mean test"

    assert np.allclose(
        out_std.flatten(), expected_std_v2
    ), "ShiftML2 failed regression variance test"

    assert np.allclose(
        out_ensemble.flatten(), expected_ensemble_v2
    ), "ShiftML2 failed regression ensemble test"
