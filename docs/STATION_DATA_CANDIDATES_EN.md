# Physical-data candidate registry — v110

[Русский](STATION_DATA_CANDIDATES_RU.md)

`atomic_model_station_evidence` provides versioned metadata for existing bundled sources: conditions, measured quantities and units, primary references, versions, publication/raw-data rights, uncertainty semantics and missing station inputs.

```bash
python -m atomic_model_station_evidence --output results/candidates.json
```

The registry works offline from the wheel. It verifies pinned artifact checksums in the installed packages and reuses their existing publications and notices. Changed files, unsupported units, incomplete metadata and unknown admission flags are refused. Existing output files are preserved.

The three candidate families are:

- [BEAVRS](https://github.com/mit-crpg/BEAVRS/tree/1ccae04dc3830a5977f08ed7f5f7e1e51ded0d22): pinned sequential cycle-1 HZP bank-worth table; not a complete station passport.
- [NIST TN 2294](https://doi.org/10.6028/NIST.TN.2294): component friction data; symbol reading bounds are not instrument uncertainties.
- [MOTEL](https://publikationen.bibliothek.kit.edu/1000187398): integral-facility publication; article licensing does not supply rights to missing raw measurements.

Records distinguish reported measurements, digitized figures and unavailable raw streams. Missing imported uncertainty information is not presented as absent from the upstream benchmark.

`admitted_for_station_validation`, `experimental_validation` and `full_r2_accepted` remain false. Integrity checks do not establish model applicability, data completeness or permission to mix plant parameters. Explicit gaps identify the geometry/boundary, coefficient, clock, covariance and independent validation work still needed for NPP-13.02/03.

Artifacts use an allowlisted source package, a portable relative path and SHA256. Absolute paths, traversal and unknown packages are rejected. Returned metadata is a fresh copy. Rights declarations describe the retained source notices; new artifacts and measurement streams require their own review. The registry neither downloads new data nor infers raw-data licenses from an article's license.

## Acceptance v110

Three candidates and 11 pinned artifacts were verified. The installed wheel reproduced the complete registry JSON and repeated its fifteen tests within the combined forty-four new-test suite. The shared targeted run passed 153 tests; the full library suite was not run. NPP-13.01 is accepted as a candidate registry with explicit gaps. The plan is 52/76 complete, with 24 open. NPP-13.02/03/04 and full R2 remain unaccepted. [Receipt](benchmarks/station_data_candidates_v110.json).
