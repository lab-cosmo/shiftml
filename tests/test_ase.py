# TODO: test for rotational invariance, translation invariance,
# and permutation, as well as size extensivity
import numpy as np
import pytest
from ase import Atoms
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

expected_output_ll_feat = {
    "ShiftML3": np.array(
        [
            -0.833786,
            3.8337648,
            -1.3120332,
            0.5230308,
            -4.0706124,
            -0.39981633,
            0.08153731,
            1.5392827,
            -0.8842108,
            -0.0541966,
            0.9843201,
            2.7937062,
            2.9484923,
            1.0625151,
            -0.20434844,
            -0.98111576,
            -0.9566989,
            0.84103,
            0.136049,
            -3.2029881,
            1.481773,
            -1.8953875,
            -2.54192,
            2.5098956,
            -2.7613125,
            3.3332195,
            -3.8492508,
            5.248315,
            1.5671709,
            4.795123,
            -0.1833263,
            0.99321324,
            0.97483873,
            0.47999394,
            -2.1559217,
            0.9834585,
            -0.53497064,
            0.06978589,
            1.2847071,
            -0.46289086,
            2.4620256,
            1.4643619,
            -0.44862294,
            -0.48347735,
            1.5859232,
            1.7806627,
            -2.3415565,
            1.5489575,
            -1.4462423,
            0.6326928,
            -1.4858731,
            1.3954905,
            4.461746,
            -2.4435005,
            -0.5386629,
            1.3182665,
            -0.87584174,
            -0.75050086,
            0.2853713,
            -2.8299348,
            -0.905771,
            -2.7950366,
            -3.672275,
            -0.34476104,
            0.4830301,
            -2.400648,
            -0.45583522,
            0.25815305,
            -1.6067216,
            5.0060463,
            -3.7211242,
            1.2728895,
            -0.8946893,
            -1.7772882,
            3.8220112,
            1.6824867,
            1.8407915,
            -0.57527,
            2.1032882,
            -0.86501306,
            -2.3451805,
            0.8962443,
            1.7138042,
            0.258034,
            -0.5085196,
            -1.0886493,
            2.1357312,
            -1.5594299,
            -0.43711087,
            -2.0931516,
            -1.3727262,
            1.4907651,
            -0.92126125,
            1.8380152,
            0.82821774,
            0.3845452,
            2.4616685,
            -0.08318162,
            -0.6842626,
            0.353562,
            2.342928,
            3.6159682,
            0.13228738,
            2.669129,
            -1.9788562,
            2.583807,
            -1.0744799,
            -1.5327199,
            -1.6303927,
            1.5039983,
            2.7896504,
            -1.1296909,
            -1.0357462,
            1.7293165,
            -0.512146,
            -2.2845469,
            4.635363,
            1.5150446,
            0.30609328,
            -1.3577303,
            -1.8782568,
            3.1361423,
            -2.168019,
            -0.59488225,
            0.57427484,
            -0.73027754,
            -0.15899932,
            0.5650684,
            -0.17604506,
            -1.1946821,
            -1.9948871,
            2.0276642,
            0.5343809,
            -0.1557374,
            -2.2142203,
            -0.7745656,
            -0.2848955,
            1.164304,
            -0.4675008,
            0.8642231,
            -3.1537433,
            -0.9718432,
            -1.405849,
            -2.4362037,
            3.0314903,
            -1.4419405,
            -1.7458878,
            0.46988344,
            0.7824265,
            1.3106066,
            -3.6510596,
            1.6114376,
            0.19771975,
            1.4362212,
            -1.4143219,
            -0.1739051,
            1.7455926,
            1.5910828,
            1.5714902,
            0.7357051,
            -3.219796,
            -2.1878529,
            1.4019806,
            -2.1862724,
            -3.8366854,
            -0.7268785,
            2.4465008,
            -1.7081892,
            -0.05461895,
            0.85107136,
            -1.303362,
            2.9121377,
            -1.1711589,
            2.1013474,
            -5.396477,
            1.8710508,
            2.110913,
            1.2154074,
            -1.6074562,
            -0.02192032,
            1.8382369,
            0.5872793,
            -2.966206,
            3.2857668,
            3.4614334,
            -1.4445789,
            -1.503231,
            -1.7323644,
            -0.06616241,
            -0.87369853,
            2.3749137,
            0.78689915,
        ],
        dtype=np.float32,
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
    cs_tensor = model.get_cs_tensor(frame, return_symmetric=True).reshape((2, 3, 3))
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


def test_shiftml3_last_layer_features():
    """Test ShiftML3 last layer features extraction"""
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML("ShiftML3", device="cpu")
    ll_feat = model.get_last_layer_features(frame)[0]

    assert ll_feat.shape == (192,), "Last layer features shape mismatch"

    assert np.allclose(
        ll_feat, expected_output_ll_feat["ShiftML3"], rtol=1e-4
    ), "Last layer features values do not match expected output"

    frame = Atoms("C", positions=[[0, 0, 0]])
    ll_feat = model.get_last_layer_features(frame)

    assert ll_feat.shape == (
        1,
        192,
    ), "Last layer features shape mismatch for single atom"

    # assert that they are equal to zero
    assert not np.any(ll_feat), "Last layer features for single atom should be zero"
