# TODO: test for rotational invariance, translation invariance,
# and permutation, as well as size extensivity
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

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


# Molecular models are trained on isolated molecules, so they are tested on a
# molecule (acetamide, no periodic boundary conditions) rather than on the
# diamond crystal used above.
MOLECULAR_MODELS = ["ShiftML4mol"]

# float32 noise is ~1e-4 ppm, which a pure relative tolerance does not cover for
# the near-zero off-diagonal tensor components; 1e-3 ppm is still negligible
MOLECULE_ATOL = 1e-3

expected_outputs_molecule = {
    "ShiftML4mol": np.array(
        [
            -67.7019207,
            7.46164007,
            146.85052892,
            166.08415287,
            26.26352101,
            30.0015234,
            29.56172587,
            29.44332809,
            26.70140958,
        ]
    ),
}

expected_outputs_molecule_tensors = {
    "ShiftML4mol": np.array(
        [
            [
                [-171.04140871, 21.71560769, 2.53911283],
                [21.71560769, -331.70150789, -4.67890532],
                [2.53911283, -4.67890532, 299.63715449],
            ],
            [
                [-69.68595407, 20.97874591, -0.46858831],
                [20.97874591, -7.97486217, -0.86945864],
                [-0.46858831, -0.86945864, 100.04573644],
            ],
            [
                [169.29153831, -95.57671048, 6.19966225],
                [-95.57671048, 110.1524049, 2.38082059],
                [6.19966225, 2.38082059, 161.10764356],
            ],
            [
                [196.10465184, 3.80338025, 3.23040823],
                [3.80338025, 157.45532921, -0.55297114],
                [3.23040823, -0.55297114, 144.69247755],
            ],
            [
                [25.72146252, -1.10734972, -0.08342639],
                [-1.10734972, 30.91681826, -1.72647311],
                [-0.08342639, -1.72647311, 22.15228224],
            ],
            [
                [32.00461749, -3.02184835, 0.84005218],
                [-3.02184835, 30.88705596, -0.90339549],
                [0.84005218, -0.90339549, 27.11289674],
            ],
            [
                [29.35519781, 0.85686115, 2.12893029],
                [0.85686115, 29.90543958, 3.13654073],
                [2.12893029, 3.13654073, 29.42454023],
            ],
            [
                [29.06039245, 0.53905812, -2.47353003],
                [0.53905812, 27.9700134, -2.62356554],
                [-2.47353003, -2.62356554, 31.29957843],
            ],
            [
                [33.293819, -0.68656343, 1.2053408],
                [-0.68656343, 23.52881053, -0.21437061],
                [1.2053408, -0.21437061, 23.28159922],
            ],
        ]
    ),
    "ShiftML4mol0": np.array(
        [
            [
                [-167.80500113, 25.39878336, -0.13790594],
                [25.39878336, -328.20240339, -0.59436489],
                [-0.13790594, -0.59436489, 295.13274869],
            ],
            [
                [-68.46968818, 19.9809658, -0.91346584],
                [19.9809658, -9.73436878, -1.03822549],
                [-0.91346584, -1.03822549, 100.09922636],
            ],
            [
                [169.43899879, -94.81195355, 6.23558578],
                [-94.81195355, 110.26942524, 3.78651468],
                [6.23558578, 3.78651468, 162.61621814],
            ],
            [
                [196.25102005, 3.30722004, 3.34919628],
                [3.30722004, 157.35108382, 0.1175999],
                [3.34919628, 0.1175999, 143.78550543],
            ],
            [
                [26.57709292, -1.92102537, 0.0961771],
                [-1.92102537, 30.00027547, -1.72004661],
                [0.0961771, -1.72004661, 20.13614794],
            ],
            [
                [32.23834433, -3.18796301, 0.80546834],
                [-3.18796301, 31.36846397, -0.3503759],
                [0.80546834, -0.3503759, 27.30998848],
            ],
            [
                [30.07531805, 0.42533908, 2.27212548],
                [0.42533908, 29.8794549, 3.0629858],
                [2.27212548, 3.0629858, 29.44012321],
            ],
            [
                [29.61604704, 0.14078778, -2.51093997],
                [0.14078778, 28.67776049, -1.82464817],
                [-2.51093997, -1.82464817, 31.00035449],
            ],
            [
                [33.76139893, -0.45714508, 1.37780669],
                [-0.45714508, 23.57767374, -0.06915687],
                [1.37780669, -0.06915687, 22.47718159],
            ],
        ]
    ),
}

expected_outputs_molecule_cs_iso_ensemble = {
    "ShiftML4mol": np.array(
        [
            [
                -66.95821861,
                -68.63035488,
                -65.44035721,
                -72.20985431,
                -66.08332188,
                -68.72648596,
                -68.2840381,
                -65.28273467,
            ],
            [
                7.2983898,
                8.1358895,
                8.10221456,
                6.75831323,
                7.0445657,
                7.23298684,
                6.6528395,
                8.46792141,
            ],
            [
                147.44154739,
                145.4926907,
                148.94909267,
                146.03611495,
                146.75122196,
                147.408564,
                146.31812997,
                146.40686974,
            ],
            [
                165.79586976,
                165.24190035,
                166.66756859,
                166.23094393,
                166.37899917,
                166.35026204,
                166.22038995,
                165.78728915,
            ],
            [
                25.57117211,
                26.6808068,
                25.72166323,
                26.38589883,
                26.56942399,
                26.89722065,
                26.01991007,
                26.26207237,
            ],
            [
                30.30559893,
                30.17653952,
                29.31015072,
                30.05915733,
                29.79361418,
                30.55416806,
                29.83289868,
                29.98005975,
            ],
            [
                29.79829872,
                29.83108609,
                29.01862346,
                29.79617559,
                29.86510902,
                29.93454461,
                28.88416593,
                29.36580358,
            ],
            [
                29.76472068,
                29.75131677,
                29.11204776,
                29.57475565,
                29.57606829,
                29.55141444,
                28.98250603,
                29.23379514,
            ],
            [
                26.60541809,
                26.75586956,
                26.23472056,
                26.5530204,
                26.85462812,
                27.10303207,
                26.44323214,
                27.06135574,
            ],
        ]
    ),
}

expected_output_molecule_ll_feat = {
    "ShiftML4mol": np.array(
        [
            0.40314084,
            0.00934121,
            1.502971,
            0.2306018,
            0.7147319,
            -0.46135873,
            -0.06526825,
            -0.00976735,
            1.2999933,
            -0.32949305,
            0.37493718,
            -0.21078396,
            0.13219061,
            -0.31761438,
            0.7908051,
            -0.22860089,
            0.04734862,
            -0.12318805,
            -0.20573248,
            -0.2502836,
            0.48471785,
            0.12814239,
            -1.058759,
            0.05575405,
            2.3433967,
            -0.0057105,
            0.20663631,
            -0.08789997,
            -0.43192357,
            -0.38366184,
            -0.36572057,
            0.31312972,
            0.12389362,
            -0.25100058,
            0.50962013,
            0.20160097,
            1.312711,
            2.662938,
            0.13940874,
            -0.73939794,
            1.0680573,
            -3.651866,
            -0.33498138,
            0.74204737,
            0.705384,
            0.5336234,
            0.10245822,
            0.18674167,
            -2.2528148,
            -0.1252688,
            0.07816873,
            0.1631313,
            0.51941717,
            0.19853291,
            0.17056985,
            -6.4659986,
            -3.318618,
            0.28793788,
            -0.02759342,
            -0.4586576,
            2.905009,
            -0.04382655,
            -0.08779775,
            0.13617906,
            -0.01071268,
            -0.08907019,
            -0.47446585,
            -0.08002738,
            -0.20485076,
            -0.37605977,
            2.7172647,
            1.6792072,
            -0.09288254,
            -0.5509335,
            -0.89036715,
            0.33611026,
            -0.07553998,
            -0.19190693,
            0.18414114,
            0.00428715,
            1.5952791,
            -0.07806966,
            2.8000865,
            -0.4602002,
            -0.0500576,
            0.36656788,
            0.0280757,
            0.16002667,
            -0.13565254,
            -0.12348524,
            0.61258876,
            -2.8488052,
            -0.21517158,
            1.6224673,
            0.3951561,
            0.14511448,
            -0.11477734,
            3.0393848,
            -0.24936754,
            0.1626049,
            -0.0521923,
            -0.7917455,
            -0.201203,
            -0.30403316,
            0.17005707,
            -0.22229645,
            2.2520053,
            -0.20129833,
            0.36325014,
            0.7835305,
            0.0191206,
            0.20502242,
            0.00025843,
            0.5100512,
            3.298244,
            0.16478956,
            0.15360922,
            0.04242456,
            -0.11764379,
            -0.3921401,
            0.05367459,
            -0.220442,
            -2.5630243,
            0.12522772,
            -1.8108952,
            -0.5765002,
            2.6558704,
            -0.17992997,
            -0.32051945,
            -0.58935934,
            1.2121181,
            -0.37702602,
            0.12154377,
            0.68831605,
            2.9586854,
            0.29723898,
            -0.44126955,
            -0.2900899,
            -0.52815133,
            -0.2344621,
            -0.22305211,
            2.5335355,
            -0.04040375,
            -0.16734059,
            -0.05218315,
            0.23546866,
            0.00282973,
            0.3779138,
            1.4106209,
            -0.28227788,
            0.37878054,
            0.19449015,
            0.28357756,
            0.00059068,
            -0.41660658,
            0.01571776,
            0.22871545,
            0.420673,
            -0.27628142,
            -1.7041563,
            0.6214978,
            3.0023203,
            0.06380898,
            2.6155066,
            -0.44759557,
            1.5682147,
            0.20095702,
            -0.08091556,
            0.04369133,
            0.13029233,
            -0.31060347,
            -0.05261085,
            -0.8597013,
            -0.30294913,
            -0.03222945,
            -0.04100758,
            0.53153807,
            -0.5358017,
            -0.10234406,
            0.6297236,
            -0.23234692,
            -0.21628281,
            -0.20004465,
            0.26393688,
            0.05026942,
            -0.16161832,
            0.9608623,
            0.09792875,
            0.5489709,
            -0.19419512,
            0.8303317,
            -0.02440679,
        ],
        dtype=np.float32,
    ),
}


def acetamide():
    return molecule("CH3CONH2")


@pytest.mark.parametrize("model_version", MOLECULAR_MODELS)
def test_molecule_regression(model_version):
    """Regression test for the isotropic shieldings of a molecule."""
    frame = acetamide()
    model = ShiftML(model_version, device="cpu")
    out = model.get_cs_iso(frame)

    assert out.shape == (len(frame),), "CS iso shape mismatch"
    assert np.allclose(
        out, expected_outputs_molecule[model_version], atol=MOLECULE_ATOL
    ), f"{model_version} failed regression test"


@pytest.mark.parametrize("model_version", MOLECULAR_MODELS)
def test_molecule_size_extensivity(model_version):
    """Two copies of a molecule far apart must each give the isolated result."""
    frame = acetamide()
    other = frame.copy()
    # well beyond the 10 A interaction range of the models
    other.translate([30.0, 0.0, 0.0])
    frame += other

    model = ShiftML(model_version, device="cpu")
    out = model.get_cs_iso(frame)

    expected = expected_outputs_molecule[model_version]
    assert np.allclose(
        out, np.concatenate([expected, expected]), atol=MOLECULE_ATOL
    ), f"{model_version} failed size extensivity test"


@pytest.mark.parametrize("model_version", MOLECULAR_MODELS)
def test_molecule_tensors(model_version):
    frame = acetamide()
    model = ShiftML(model_version, device="cpu")
    cs_tensor = model.get_cs_tensor(frame, return_symmetric=True)
    assert cs_tensor.shape == (len(frame), 3, 3), "CS tensor shape mismatch"

    assert np.allclose(
        cs_tensor, cs_tensor.transpose(0, 2, 1)
    ), "CS tensor is not symmetric"

    assert np.allclose(
        cs_tensor,
        expected_outputs_molecule_tensors[model_version],
        rtol=1e-4,
        atol=MOLECULE_ATOL,
    ), "CS tensor values do not match expected output"


@pytest.mark.parametrize("model_version", MOLECULAR_MODELS)
def test_molecule_cs_iso_ensemble(model_version):
    """Regression test for the per-member isotropic shieldings of a molecule."""
    frame = acetamide()
    model = ShiftML(model_version, device="cpu")
    cs_iso_ensemble = model.get_cs_iso_ensemble(frame)

    n_members = len(resolve_ensemble_members[model_version])
    assert cs_iso_ensemble.shape == (
        len(frame),
        n_members,
    ), "CS iso ensemble shape mismatch"

    assert np.allclose(
        cs_iso_ensemble,
        expected_outputs_molecule_cs_iso_ensemble[model_version],
        rtol=1e-4,
        atol=MOLECULE_ATOL,
    ), "CS iso ensemble values do not match expected output"

    # the ensemble mean is what get_cs_iso returns
    assert np.allclose(
        cs_iso_ensemble.mean(axis=-1), model.get_cs_iso(frame), rtol=1e-4
    ), "CS iso is not the mean over the ensemble members"

    # guard against a model being uploaded twice to zenodo
    for i in range(n_members):
        for j in range(i + 1, n_members):
            assert not np.array_equal(
                cs_iso_ensemble[:, i], cs_iso_ensemble[:, j]
            ), f"{model_version} members {i} and {j} predict identical values"


@pytest.mark.parametrize("model_version", ["ShiftML4mol0"])
def test_molecule_single_model_tensors(model_version):
    """Regression test of one of the ensemble members (model 0)"""
    frame = acetamide()
    model = ShiftML(model_version, device="cpu")
    cs_tensor = model.get_cs_tensor(frame, return_symmetric=True).reshape((-1, 3, 3))
    assert cs_tensor.shape == (len(frame), 3, 3), "CS tensor shape mismatch"

    assert np.allclose(
        cs_tensor, cs_tensor.transpose(0, 2, 1)
    ), "CS tensor is not symmetric"

    assert np.allclose(
        cs_tensor,
        expected_outputs_molecule_tensors[model_version],
        rtol=1e-3,
        atol=MOLECULE_ATOL,
    ), "CS tensor values do not match expected output"


@pytest.mark.parametrize("model_version", MOLECULAR_MODELS)
def test_molecule_fail_potassium(model_version):
    """Unlike ShiftML3 and ShiftML4, the molecular models are not fitted on K"""
    frame = Atoms("KCl", positions=[[0, 0, 0], [0, 0, 2.7]])
    model = ShiftML(model_version, device="cpu")
    with pytest.raises(
        ValueError, match="Model is fitted only for the following atomic numbers:"
    ):
        model.get_cs_iso(frame)


@pytest.mark.parametrize("model_version", MOLECULAR_MODELS)
def test_molecule_last_layer_features(model_version):
    """Test last layer features extraction on a molecule"""
    frame = acetamide()
    model = ShiftML(model_version, device="cpu")
    ll_feat = model.get_last_layer_features(frame)

    assert ll_feat.shape == (len(frame), 192), "Last layer features shape mismatch"

    assert np.allclose(
        ll_feat[0],
        expected_output_molecule_ll_feat[model_version],
        rtol=1e-3,
        atol=1e-4,
    ), "Last layer features values do not match expected output"
