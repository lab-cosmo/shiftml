# Changelog

All notable changes to ShiftML are documented in this file.

## [0.2.0] - 2026-08-24

### Added

- **New model `ShiftML4`.** Trained on PBE0 molecular-corrected GIPAW-PBE data
  (1.2 million chemical shieldings from 12600 organic crystals), covering the
  same 12 elements as `ShiftML3`: H, C, N, O, S, F, P, Cl, Na, Ca, Mg and K.
  Against a hold-out set it reaches isotropic shielding RMSEs of 0.40 ppm for
  $^{1}\text{H}$ and 2.22 ppm for $^{13}\text{C}$, compared to 0.42 ppm and
  2.24 ppm for `ShiftML3` on the same structures computed at GIPAW-PBE level.
  Select it with `ShiftML("ShiftML4")`.
- `shiftml.utils.loading.load_model`, which rebuilds a model's metatomic
  wrapper with the installed metatomic. This makes the published model files
  independent of the `metatomic-torch` release that exported them, so a new
  metatomic no longer requires re-uploading models or releasing ShiftML.

### Changed

- **Dependencies were narrowed and modernised.** `MetatomicCalculator` moved
  out of `metatomic-torch` into its own `metatomic-ase` package, and ShiftML
  follows it. `metatensor-learn` and `metatensor-operations` are no longer
  required at run time: they are only needed to *train* models, and everything
  the published models need from them is already compiled into the TorchScript
  archives. Dropping them removes the main source of version conflicts with
  other metatensor-ecosystem packages such as `metatrain` and `uPET`.

  | package | 0.1.1 | 0.2.0 |
  | --- | --- | --- |
  | `python` | `>=3.9` | `>=3.10` |
  | `metatensor-torch` | `>=0.7.6,<0.9` | `>=0.10,<0.11` |
  | `metatomic-torch` | `>=0.1.2,<0.2` | `>=0.1.17,<0.2` |
  | `metatomic-ase` | — | `>=0.1.1,<0.2` |
  | `metatensor-learn` | `>=0.3.2,<0.4` | removed |
  | `metatensor-operations` | `>=0.3.3,<0.4` | removed |



### Migration notes

- Requires Python 3.10 or newer.
- If you pin `metatensor-torch` or `metatomic-torch`, update those pins; the
  new ranges are not compatible with the 0.1.1 ones.

## [0.1.1] - 2026-02-17

- Retry Zenodo downloads that fail with HTTP 423.
- Documentation and README fixes.

## [0.1.0] - 2025-06-15

- First public release.

[0.2.0]: https://github.com/lab-cosmo/ShiftML/releases/tag/v0.2.0
[0.1.1]: https://github.com/lab-cosmo/ShiftML/releases/tag/v0.1.1
[0.1.0]: https://github.com/lab-cosmo/ShiftML/releases/tag/v0.1.0
