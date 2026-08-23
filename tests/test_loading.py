"""Tests for building a modern AtomisticModel out of a ShiftML model file."""

import os

import numpy as np
import pytest
import torch
from ase.build import bulk
from metatomic.torch import AtomisticModel, ModelOutput
from metatomic_ase import MetatomicCalculator
from platformdirs import user_cache_path

from shiftml.ase import ShiftML
from shiftml.ase.calculator import ShiftML_model, resolve_ensemble_members
from shiftml.utils.loading import load_model

#: the ensembles
ENSEMBLES = sorted(resolve_ensemble_members)

#: the first committee member of each ensemble. ShiftML4 starts at 1, because
#: model_0 on Zenodo is a duplicate upload of model_1.
MEMBERS = [
    version + str(members[0]) for version, members in resolve_ensemble_members.items()
]


def _model_file(model_version):
    """Path of a cached model file, downloading it if needed."""
    ShiftML_model(model_version, device="cpu")
    return os.path.join(
        os.path.expanduser(os.path.join(user_cache_path(), "shiftml", model_version)),
        model_version + ".pt",
    )


@pytest.mark.parametrize("model_version", MEMBERS)
def test_load_model_builds_a_usable_atomistic_model(model_version):
    """The published files cannot be opened by load_atomistic_model; ours can."""
    model = load_model(_model_file(model_version))

    assert isinstance(model, AtomisticModel)
    # the neighbor list request must survive the re-wrap, otherwise evaluation
    # dies with "AssertionError: no neighbor list found"
    assert [options.cutoff for options in model.requested_neighbor_lists()] == [5.0]
    assert set(model.capabilities().outputs.keys()) >= {
        "mtt::cs_iso",
        "mtt::aux::cs_iso_last_layer_features",
    }


@pytest.mark.parametrize("model_version", MEMBERS)
def test_load_model_is_deterministic(model_version):
    """Two independent loads of the same file must predict identically."""
    path = _model_file(model_version)
    frame = bulk("C", "diamond", a=3.566).repeat((2, 1, 1))
    outputs = {"mtt::cs_iso": ModelOutput(quantity="", unit="", sample_kind="atom")}

    predictions = []
    for _ in range(2):
        calculator = MetatomicCalculator(load_model(path), device="cpu")
        out = calculator.run_model(frame, outputs)["mtt::cs_iso"]
        out = out.components_to_properties(["o3_mu"])
        predictions.append(
            np.concatenate(
                [block.values.to("cpu").numpy() for block in out.blocks()], axis=1
            )
        )

    assert np.array_equal(predictions[0], predictions[1])


def test_load_model_rejects_non_metatomic_archives(tmp_path):
    path = str(tmp_path / "not-a-model.pt")
    torch.jit.script(torch.nn.Linear(2, 2)).save(path)

    # the archive carries none of the metadata metatomic writes, so it is
    # rejected before we ever look for an AtomisticModel inside it
    with pytest.raises(ValueError, match="metatomic model"):
        load_model(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
@pytest.mark.parametrize("model_version", ENSEMBLES)
def test_cpu_and_cuda_agree(model_version):
    """The same structure must give the same answer on both devices."""
    frame = bulk("C", "diamond", a=3.566).repeat((2, 2, 2))

    on_cpu = ShiftML(model_version, device="cpu").get_cs_iso(frame)
    on_cuda = ShiftML(model_version, device="cuda").get_cs_iso(frame)

    # float32 summation order differs between the two neighbor-list backends, so
    # this is a "same answer for all practical purposes" check, not bit equality
    assert np.allclose(on_cpu, on_cuda, atol=1e-2)
