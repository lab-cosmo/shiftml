"""Loading ShiftML models across metatomic versions.

A ShiftML model file is a TorchScript archive containing the scripted network
*and* a thin :py:class:`metatomic.torch.AtomisticModel` wrapper around it, built
by whichever ``metatomic-torch`` was installed when the model was exported.

The network ages well: every TorchScript class and operator it uses
(``metatensor.TensorMap``, ``metatomic.System``, ...) keeps a compatible schema,
so ``torch.jit.load`` opens the 2025 archives under current releases without
complaint.  The *wrapper* does not: recent ``metatomic-torch`` expects
attributes on it that did not exist back then, and
:py:func:`metatomic.torch.load_atomistic_model` therefore fails with::

    AttributeError: 'RecursiveScriptModule' object has no attribute
                    '_model_capabilities_outputs_names'

So ShiftML never uses the wrapper that came with the file.  It takes the network
out and builds a fresh wrapper with the installed metatomic, every time a model
is loaded.  Nothing is read from or written to the model file beyond
``torch.jit.load``, and the network is reused verbatim, so predictions are
exactly the ones the file encodes.

This is deliberately *not* conditional on the version that exported the file.
Any future ``AtomisticModel`` change breaks every previously exported model, no
matter how recently it was written, and re-wrapping is the fix in all of those
cases too.  Always re-wrapping costs about 0.04 s per model over a plain load
and removes a whole class of "which release wrote this file" bugs.
"""

import warnings
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelOutput,
    NeighborListOptions,
    System,
    load_model_extensions,
)

__all__ = ["WrappedModel", "load_model"]


class WrappedModel(torch.nn.Module):
    """Carries a scripted ShiftML network plus its neighbor list request.

    An exported ``AtomisticModel`` keeps the requested neighbor lists on the
    *wrapper* (``_requested_neighbor_lists``), not on the network it wraps:
    TorchScript only serialises the methods reachable from ``forward()``, so the
    network's own ``requested_neighbor_lists()`` is not part of the archive.

    Building a new ``AtomisticModel`` straight around the bare network would
    therefore produce a model that requests no neighbor list at all, and that
    fails at evaluation time with ``AssertionError: no neighbor list found``.
    This wrapper puts the request back where metatomic looks for it.

    It does nothing else: ``forward()`` is a plain delegation.
    """

    _requested_nl: List[NeighborListOptions]

    def __init__(
        self, module: torch.nn.Module, requested_nl: List[NeighborListOptions]
    ):
        super().__init__()
        self.module = module
        self._requested_nl = requested_nl

    @torch.jit.export
    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return self._requested_nl

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        return self.module(systems, outputs, selected_atoms)


def load_model(path: str, extensions_directory: Optional[str] = None) -> AtomisticModel:
    """Load a ShiftML model file, wrapped for the installed metatomic.

    A drop-in replacement for
    :py:func:`metatomic.torch.load_atomistic_model` that additionally accepts
    model files written by older ``metatomic-torch`` releases -- which is what
    every currently published ShiftML model is.

    :param path: path to a ShiftML ``.pt`` model file
    :param extensions_directory: directory containing the TorchScript extensions
        required by the model, if any
    """
    load_model_extensions(
        path, str(extensions_directory) if extensions_directory is not None else None
    )

    exported = torch.jit.load(path)

    original_name = getattr(exported, "original_name", None)
    if original_name != "AtomisticModel":
        raise ValueError(
            f"the file at '{path}' does not contain an exported metatomic model "
            f"(found a '{original_name}' TorchScript module)"
        )

    network = exported.module
    network.eval()

    wrapper = WrappedModel(network, list(exported._requested_neighbor_lists))
    wrapper.eval()

    with warnings.catch_warnings():
        # the published models declare a ``features`` output; metatomic renamed
        # it to ``feature`` and handles both names transparently.  The name is
        # baked into the model file, so there is nothing our users could do
        # about this warning.
        warnings.filterwarnings("ignore", message=".*output name is deprecated.*")

        return AtomisticModel(
            torch.jit.script(wrapper),
            exported.metadata(),
            exported.capabilities(),
        )
