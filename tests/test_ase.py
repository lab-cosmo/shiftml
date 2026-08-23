# TODO: test for rotational invariance, translation invariance,
# and permutation, as well as size extensivity
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from shiftml.ase import ShiftML
from shiftml.ase.calculator import resolve_ensemble_members

expected_outputs = {
    "ShiftML3": np.array([70.39079043, 70.4060931]),
    "ShiftML4": np.array([96.64889468, 97.20188181]),
}

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
    "ShiftML4": np.array(
        [
            [
                [96.85077443, -3.0357784, -4.56176351],
                [-3.0357784, 98.44419061, -1.0511704],
                [-4.56176351, -1.0511704, 94.65171898],
            ],
            [
                [97.36443632, -2.45374251, -4.2977706],
                [-2.45374251, 98.86575493, -1.27111783],
                [-4.2977706, -1.27111783, 95.37545419],
            ],
        ]
    ),
    "ShiftML41": np.array(
        [
            [
                [106.07212586, -7.52502376, -4.40938846],
                [-7.52502376, 117.40121492, -8.17602736],
                [-4.40938846, -8.17602736, 108.79979117],
            ],
            [
                [108.98967731, -7.40468743, -4.53289187],
                [-7.40468743, 120.79024057, -10.52324604],
                [-4.53289187, -10.52324604, 111.35247847],
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
    ),
    "ShiftML4": np.array(
        [
            [
                110.75771065,
                116.96930041,
                78.62527331,
                126.99642698,
                93.63963989,
                36.13712648,
                113.41678499,
            ],
            [
                113.71079878,
                117.13290472,
                78.694438,
                127.06369759,
                93.47083788,
                36.7736733,
                113.56682241,
            ],
        ]
    ),
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
    ),
    "ShiftML4": np.array(
        [
            0.5609825,
            2.7172568,
            -0.2877852,
            -2.184326,
            0.69747245,
            1.224881,
            1.96371,
            1.1648024,
            -1.5328699,
            1.4358202,
            -2.67571,
            0.4286151,
            -1.4513013,
            -0.00947646,
            -2.448284,
            -1.1643007,
            1.7976757,
            -0.41772932,
            1.2626259,
            -1.0997202,
            0.6980895,
            3.476351,
            -1.993495,
            -1.603404,
            -1.8804408,
            1.2517653,
            1.8202511,
            -2.9050562,
            -1.818648,
            1.8017226,
            3.4879706,
            0.57003593,
            0.02294639,
            1.8398455,
            -1.1948487,
            -0.15684387,
            0.02379738,
            1.9590023,
            -2.943575,
            -1.3291196,
            -0.15528461,
            1.4327166,
            -2.6197124,
            -1.260122,
            -1.2199229,
            2.4372013,
            -1.2791785,
            -0.83276856,
            -0.7957949,
            1.7635416,
            0.3861704,
            0.2581546,
            -3.1573355,
            1.0968258,
            -2.70159,
            1.2474861,
            0.06080331,
            1.8160388,
            5.250097,
            -0.646122,
            2.4826846,
            2.321469,
            -1.8388216,
            0.8493403,
            -1.286049,
            0.23613875,
            -2.4943995,
            -3.8924496,
            2.9062877,
            1.2949632,
            -0.81811476,
            0.8265231,
            2.2870865,
            -0.5593248,
            -0.25755468,
            -1.4845562,
            -2.515249,
            3.5113537,
            -2.1467674,
            1.0289403,
            0.26602185,
            -2.29845,
            0.14669003,
            -1.745042,
            -1.7547624,
            2.1772492,
            3.1875,
            1.2409151,
            0.4121313,
            -3.928384,
            0.6138999,
            -1.8740951,
            0.56199795,
            -0.21587221,
            2.3386557,
            -1.875455,
            -3.9023604,
            0.33731937,
            0.51424223,
            3.8338406,
            0.31544456,
            -3.6975858,
            0.9740005,
            -0.4960635,
            1.7996023,
            -2.2457736,
            0.90574026,
            0.48627016,
            -0.42695808,
            0.28299555,
            0.50194484,
            -1.1265019,
            1.5202895,
            -0.8755849,
            2.1213164,
            -3.2362905,
            -0.6852894,
            -2.2612538,
            -1.7326864,
            -1.2749962,
            2.3432746,
            -1.1327437,
            1.7813282,
            -1.1808871,
            -0.85365474,
            -0.06679466,
            0.6423766,
            -1.5178019,
            -3.9741275,
            1.7487878,
            -2.2107108,
            0.7179121,
            3.0379379,
            -4.7341375,
            -2.0121598,
            -0.7929834,
            2.3139746,
            0.46207577,
            0.31054285,
            -2.622459,
            2.0132356,
            0.42263696,
            -1.9220926,
            -3.7250633,
            -0.80735224,
            0.9803001,
            1.0274355,
            -2.3097692,
            -3.8272228,
            2.372414,
            -0.88321495,
            1.6486676,
            -1.478035,
            1.7919647,
            -2.0753734,
            -2.1555984,
            1.9121484,
            -3.5307746,
            -0.9733287,
            -0.19253857,
            -0.5609595,
            0.26890165,
            0.7109786,
            2.7552617,
            0.798691,
            -2.1650026,
            -2.9551065,
            0.5118226,
            -0.54392976,
            2.854548,
            0.61760014,
            -1.1296579,
            -0.9485795,
            1.65296,
            -0.54690534,
            2.127556,
            1.5512067,
            4.7153425,
            -0.65564454,
            1.1271966,
            1.1976011,
            2.7976863,
            2.3150752,
            -2.9805737,
            -2.011856,
            -1.9681007,
            -1.7609714,
            -0.8324414,
            -0.12800367,
            1.5711534,
            -0.4451204,
            1.083728,
        ],
        dtype=np.float32,
    ),
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


@pytest.mark.parametrize("model_version", ["ShiftML3", "ShiftML4"])
def test_tensors(model_version):
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(model_version, device="cpu")
    cs_tensor = model.get_cs_tensor(frame, return_symmetric=True)
    assert cs_tensor.shape == (2, 3, 3), "CS tensor shape mismatch"

    # assert that the tensor is symmetric
    assert np.allclose(
        cs_tensor, cs_tensor.transpose(0, 2, 1)
    ), "CS tensor is not symmetric"

    assert np.allclose(
        cs_tensor, expected_outputs_tensors[model_version], rtol=1e-4
    ), "CS tensor values do not match expected output"


@pytest.mark.parametrize("model_version", ["ShiftML3", "ShiftML4"])
def test_cs_iso_ensemble(model_version):
    """Regression test for the per-member isotropic shieldings of the ensemble."""
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(model_version, device="cpu")
    cs_iso_ensemble = model.get_cs_iso_ensemble(frame)

    n_members = len(resolve_ensemble_members[model_version])
    assert cs_iso_ensemble.shape == (
        2,
        n_members,
    ), "CS iso ensemble shape mismatch"

    assert np.allclose(
        cs_iso_ensemble, expected_outputs_cs_iso_ensemble[model_version], rtol=1e-4
    ), "CS iso ensemble values do not match expected output"

    # the ensemble mean is what get_cs_iso returns
    assert np.allclose(
        cs_iso_ensemble.mean(axis=-1), model.get_cs_iso(frame), rtol=1e-4
    ), "CS iso is not the mean over the ensemble members"


@pytest.mark.parametrize("model_version", ["ShiftML3", "ShiftML4"])
def test_ensemble_members_are_distinct(model_version):
    """Guard against a duplicated model being uploaded twice to zenodo.

    ShiftML4 model_0 and model_1 are the same exported model, which is why
    model_0 is left out of the ensemble; double-counting a member biases the
    ensemble mean and understates its spread.
    """
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(model_version, device="cpu")
    cs_iso_ensemble = model.get_cs_iso_ensemble(frame)

    n_members = cs_iso_ensemble.shape[-1]
    for i in range(n_members):
        for j in range(i + 1, n_members):
            assert not np.array_equal(
                cs_iso_ensemble[:, i], cs_iso_ensemble[:, j]
            ), f"{model_version} members {i} and {j} predict identical values"


@pytest.mark.parametrize("model_version", ["ShiftML30", "ShiftML41"])
def test_single_model_tensors(model_version):
    """Regression test of one of the ensemble members (model 0)"""
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(model_version, device="cpu")
    cs_tensor = model.get_cs_tensor(frame, return_symmetric=True).reshape((2, 3, 3))
    assert cs_tensor.shape == (2, 3, 3), "CS tensor shape mismatch"

    # assert that the tensor is symmetric
    assert np.allclose(
        cs_tensor, cs_tensor.transpose(0, 2, 1)
    ), "CS tensor is not symmetric"

    assert np.allclose(
        cs_tensor, expected_outputs_tensors[model_version], rtol=1e-3
    ), "CS tensor values do not match expected output"


@pytest.mark.parametrize("model_version", ["ShiftML3", "ShiftML4"])
def test_fail_invalid_species(model_version):
    """Test ShiftML models for non-fitted species"""

    frame = bulk("Si", "diamond", a=3.566)
    model = ShiftML(model_version, device="cpu")
    with pytest.raises(ValueError) as exc_info:
        model.get_cs_iso(frame)

    assert exc_info.type == ValueError
    assert "Model is fitted only for the following atomic numbers:" in str(
        exc_info.value
    )


@pytest.mark.parametrize("model_version", ["ShiftML3", "ShiftML4"])
def test_last_layer_features(model_version):
    """Test last layer features extraction"""
    frame = bulk("C", "diamond", a=3.566)
    model = ShiftML(model_version, device="cpu")
    ll_feat = model.get_last_layer_features(frame)[0]

    assert ll_feat.shape == (192,), "Last layer features shape mismatch"

    # atol floors the tolerance: some features are close to zero, where a pure
    # relative tolerance is tighter than the ~1e-5 float32 noise between platforms
    assert np.allclose(
        ll_feat, expected_output_ll_feat[model_version], rtol=1e-3, atol=1e-4
    ), "Last layer features values do not match expected output"

    frame = Atoms("C", positions=[[0, 0, 0]])
    ll_feat = model.get_last_layer_features(frame)

    assert ll_feat.shape == (
        1,
        192,
    ), "Last layer features shape mismatch for single atom"

    # assert that they are equal to zero
    assert not np.any(ll_feat), "Last layer features for single atom should be zero"
