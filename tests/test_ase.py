# TODO: test for rotational invariance, translation invariance,
# and permutation, as well as size extensivity
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from shiftml.ase import ShiftML

expected_outputs = {
    "ShiftML3": np.array([70.39079043, 70.4060931]),
    "ShiftML4": np.array([98.41249667, 99.26549643]),
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
                [98.00344336, -3.59693407, -4.54271663],
                [-3.59693407, 100.81381865, -1.94177752],
                [-4.54271663, -1.94177752, 96.42022801],
            ],
            [
                [98.81759144, -3.07261062, -4.32716075],
                [-3.07261062, 101.60631563, -2.42763386],
                [-4.32716075, -2.42763386, 97.37258223],
            ],
        ]
    ),
    "ShiftML40": np.array(
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
            -0.14657474,
            4.1477027,
            -0.43802768,
            -1.8492866,
            0.7894825,
            1.1183993,
            3.3257508,
            0.94709885,
            -1.2212133,
            0.33292234,
            -2.366565,
            -0.66327846,
            -2.8468082,
            -1.0418658,
            -2.90477,
            -1.0951183,
            2.5801547,
            0.07158357,
            0.5390599,
            -1.2790921,
            0.09342641,
            2.4678364,
            -1.4109763,
            -2.0807161,
            -1.5944364,
            0.7491117,
            1.003108,
            -3.0448308,
            -1.6345243,
            1.4602075,
            3.5445173,
            0.46086466,
            0.12402397,
            1.5118012,
            -2.170526,
            0.03086674,
            0.61968356,
            2.833278,
            -2.3961039,
            -1.0943527,
            -0.05809921,
            2.499737,
            -2.3892887,
            -1.0679858,
            -1.9865539,
            2.7897227,
            -1.8767195,
            -0.9806984,
            -0.73087615,
            2.19638,
            0.5287378,
            -0.3770408,
            -4.7328587,
            1.0721469,
            -2.9024553,
            1.1680353,
            -0.33250678,
            1.2547868,
            6.0254736,
            -0.35607314,
            2.8757873,
            1.9729389,
            -1.061392,
            -0.42363387,
            -0.6635716,
            1.4736562,
            -2.3945653,
            -3.749687,
            2.4378362,
            0.95379394,
            -0.22728574,
            1.786251,
            2.7869413,
            -0.29391408,
            -1.0288029,
            -1.9957005,
            -2.7297719,
            2.825224,
            -3.3197389,
            1.300423,
            0.30788848,
            -3.3438933,
            -0.07095826,
            -0.67488444,
            -1.9676137,
            1.9319546,
            4.534929,
            1.8756504,
            0.24524039,
            -4.898367,
            -0.19576323,
            -1.7610809,
            -0.5995854,
            -1.034543,
            2.3760774,
            -0.13361347,
            -3.3210235,
            -0.32162824,
            0.23661542,
            3.7702265,
            -0.41921484,
            -3.1049519,
            0.66661036,
            0.21874332,
            2.2206597,
            -2.6132727,
            0.46072507,
            0.3380831,
            0.56141996,
            1.3913192,
            0.22064388,
            -2.250432,
            1.6631358,
            -0.43642056,
            3.081749,
            -2.1336045,
            -0.30045778,
            -1.866585,
            -1.2027112,
            -1.9905217,
            3.429913,
            -0.9851373,
            1.4070882,
            -1.6181802,
            -0.6867299,
            0.10397542,
            0.45726752,
            -2.0976677,
            -4.412122,
            1.7227665,
            -1.835269,
            1.2522194,
            3.240079,
            -4.1101913,
            -2.4437213,
            -0.81970406,
            2.65398,
            0.9930269,
            0.5907702,
            -2.0550363,
            2.2053862,
            0.39424017,
            -1.443902,
            -5.1620245,
            -1.2094619,
            0.71800625,
            0.8068197,
            -4.1732855,
            -4.387228,
            2.7534032,
            -0.7508154,
            2.548371,
            -2.4446247,
            2.341246,
            -2.063978,
            -2.2683594,
            2.4450874,
            -3.7468784,
            -1.8795328,
            -0.40640098,
            -0.79098547,
            0.14959347,
            0.2091471,
            2.6447752,
            1.5102959,
            -1.4903622,
            -3.5459523,
            1.4629864,
            -0.4080445,
            4.287951,
            -0.62734133,
            -0.9940436,
            -0.9389037,
            1.8603239,
            -0.5584636,
            3.01642,
            2.3787026,
            3.9750974,
            -1.2064672,
            0.8007752,
            1.6945411,
            3.3230915,
            2.682377,
            -3.9585886,
            -2.236689,
            -1.827419,
            -0.9818657,
            -1.0464447,
            0.2543915,
            1.6267118,
            -0.3986627,
            1.9429059,
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

    assert cs_iso_ensemble.shape == (2, 8), "CS iso ensemble shape mismatch"

    assert np.allclose(
        cs_iso_ensemble, expected_outputs_cs_iso_ensemble[model_version], rtol=1e-4
    ), "CS iso ensemble values do not match expected output"

    # the ensemble mean is what get_cs_iso returns
    assert np.allclose(
        cs_iso_ensemble.mean(axis=-1), model.get_cs_iso(frame), rtol=1e-4
    ), "CS iso is not the mean over the ensemble members"


@pytest.mark.parametrize("model_version", ["ShiftML30", "ShiftML40"])
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

    assert np.allclose(
        ll_feat, expected_output_ll_feat[model_version], rtol=1e-3
    ), "Last layer features values do not match expected output"

    frame = Atoms("C", positions=[[0, 0, 0]])
    ll_feat = model.get_last_layer_features(frame)

    assert ll_feat.shape == (
        1,
        192,
    ), "Last layer features shape mismatch for single atom"

    # assert that they are equal to zero
    assert not np.any(ll_feat), "Last layer features for single atom should be zero"
