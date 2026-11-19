# Station capability passport and configuration migration

[Русский](STATION_CAPABILITIES_RU.md)

`atomic_model_station_capabilities` ships in the wheel. It provides a JSON passport for `StationEnv` and source-bound configuration transport without requiring a checkout.

```bash
python -m atomic_model_station_capabilities passport --output results/passport.json
python -m atomic_model_station_capabilities export-config --output results/config.json
python -m atomic_model_station_capabilities validate results/config.json
```

The passport lists interfaces, operator actions, observation visibility, default settings, implemented EOS chart bounds and heat-transfer models. Reading it does not simulate the station. API availability, valid inputs and numerical qualification of a particular recipe are separate properties. Chart bounds do not define a qualified operating range for the whole station.

## Evidence and scope

The wheel embeds an index of the published nominal requalification v105 and installed API tutorials v106. Each record contains the published receipt checksum and the scientific dependency fingerprint used by that recipe. `current_source_matches` compares the fingerprint with the current implementation.

A stale record remains visible with a false compatibility flag. Corrupted indexes, unexpected records and source changes during passport generation are refused. The index is a project publication with checksums, not an independent review or a third-party digital signature. Full mathematical evidence is described in the linked repository protocols and is not downloaded when generating the passport.

`current_source_matches` checks source compatibility only. It does not verify Python or NumPy/SciPy versions, hardware or the results of a new execution. A different runtime environment still requires replaying the applicable scientific protocol.

Qualification applies only to the declared nominal recipe. Tutorials demonstrate API use and do not qualify physical scenarios. `experimental_validation`, `industrial_plant_qualified` and `full_r1_accepted` remain false. Full coastdown, the mandatory R1 matrix and experimental validation remain explicit limitations.

## Configuration transport

`station-env-config/v2` stores every `EnvConfig` field, the station source fingerprint and a content checksum. Validation uses the environment and fault validators. It does not run a simulation or qualify an operating point.

All fields are explicit. Unknown fields are rejected rather than dropped; missing fields do not acquire new implicit defaults. Current parameter limits and correlation names remain enforced.

To wrap a complete JSON dictionary produced by `dataclasses.asdict(config)`:

```bash
python -m atomic_model_station_capabilities migrate old-config.json --from-format envconfig-raw/v1 --output results/config.json
```

The new configuration is written to the output file. A JSON audit is printed to stdout, including the input fingerprint, old/new source binding and the requalification flag. Numerical settings are preserved. Any canonical reordering of independent fault specifications is reported separately.

A v2 configuration bound to older scientific sources is rejected by ordinary loading. Explicitly rebind the settings with:

```bash
python -m atomic_model_station_capabilities migrate previous-config.json --from-format station-env-config/v2 --allow-rebind --output results/rebound-config.json
```

Rebinding sets `requalification_required=true`. It validates settings against the current schema; it does not repeat scientific qualification or migrate physical state, controller memory or command history. Checkpoints are not configuration files. Restore a checkpoint using its matching source version and the ordinary restore API.

Output files must be new, and symlink output paths are refused. JSON input is limited to 1 MiB. Migration preserves the original input file.

## Python API

```python
from dataclasses import asdict
from atomic_model_station_env import EnvConfig, StationEnv
from atomic_model_station_capabilities import passport, migrate_config, load_config, RAW_FORMAT

report = passport()
original = EnvConfig(max_steps=3, tick_s=.02, automatic=False)
document, audit = migrate_config(asdict(original), from_format=RAW_FORMAT)
env = StationEnv(load_config(document))
observation, info = env.reset()
```

## Delivery acceptance v107

Sixty targeted tests passed, including twenty new tests. A wheel installed outside the checkout repeated the twenty new tests and all eight scientific JSON documents exactly. CLI checks cover passport export, configuration export and validation, raw migration, stale-source refusal, implicit-rebind refusal and explicit rebinding. Actual station transitions and a restored checkpoint exactly matched the original configuration.

The scientific backend is unchanged; both embedded published recipe fingerprints match current sources. The full library suite and long physical scenarios were not rerun in this stage. NPP-16.04 is accepted for the station passport and documented configuration/reporting contract. The plan stands at 50/76 items, with 26 open; full R1 remains unaccepted. [Machine-readable receipt](benchmarks/station_capabilities_v107.json).
