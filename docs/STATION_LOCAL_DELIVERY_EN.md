# Local station delivery verification

[Русский](STATION_LOCAL_DELIVERY_RU.md). This protocol joins source tests, a clean sdist → wheel build, byte-for-byte wheel inspection, installation outside the checkout and native station checks. No CI, cloud runner or separate physical machine is required.

## Run

From the repository root, using an environment with scientific and development dependencies installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.verify_local_library \
  --output artifacts/my-station-delivery \
  --suite library --phase-timeout-s 14400 \
  --station-nominal docs/benchmarks/station_nominal_requalification_v105.json
```

Use a fresh output directory without symlinks in its path. The timeout applies to each execution phase, not the simulated interval. A timeout is a failure. Execution can take a long time; keep code, shipped resources, dependencies and build rules unchanged until it finishes.

After adding top-level Python packages, refresh the editable installation with `python -m pip install --no-deps --no-build-isolation -e .`. Pytest can discover `src` through its configuration; a normal new Python process still needs the project installed correctly.

## Resumable full library lane

Long checks can execute by file while retaining completed results:

```bash
python -m scripts.verify_local_tests --output artifacts/my-source-tests --suite library --batch-files 1
python -m scripts.verify_local_library --output artifacts/my-station-delivery \
  --suite library --phase-timeout-s 1100 \
  --test-receipts artifacts/my-source-tests \
  --station-nominal docs/benchmarks/station_nominal_requalification_v105.json
```

The first command prepares a plan and executes every batch. Repeating it with the same directory resumes unfinished work. Independent batches may also be run using `--prepare`, then `--batch N`; run each index in only one process at a time. `--accept` validates all receipts and produces the summary. Each batch is a local pytest process; there is no network runner.

The plan stores exact node IDs and file partitioning. Every batch and final acceptance checks source code, tests, evidence inputs, Python and installed dependency versions. BLAS/OMP use one thread and `PYTHONHASHSEED=0`; hidden filters from `PYTEST_ADDOPTS` and `PYTEST_PLUGINS` are removed. Each receipt binds the plan to byte hashes of its log, actual collected node IDs and pytest result. Library skips, missing or duplicate coverage and changed collections prevent acceptance.

Interrupted attempts remain available and can be rerun in the unchanged environment. A successful retry cannot conceal a confirmed earlier test failure: that history is rejected. Code or dependency changes require a fresh plan. The summary reports attempt counts and summed completed pytest-process time; this is not parallel makespan.

`--test-receipts` revalidates the full plan and every result before reusing those executions instead of another expensive monolithic pytest run. Without the option, the original monolithic mode remains available.

The [v115 batch-protocol receipt](benchmarks/local_test_batches_v115.json) records all 3367 test IDs, 170 receipt hashes and source/runtime fingerprints. The final library lane passed without errors or skips; incomplete monolithic attempts are retained separately.

`portable` selects the whole `tests/` tree except tests explicitly marked `external_integration`, `long_running` or `hardware_specific`. `library` selects the listed library tests; `all` applies no marker exclusions. Reports retain the profile and actual passed, failed and skipped counts. A pytest run executing no tests is rejected. Portable verification does not cover every external runtime or hardware accelerator.

## Checks

1. Validate the supplied nominal receipt, freeze shipped code/resources and packaging rules, run local tests and retain the pytest receipt.
2. Execute eight short native station scenarios. Seven must complete; the invalid command must fail specifically with `invalid_action`. Each attempt includes a checkpoint and real restoration. An additional feed-pump midpoint restart must reproduce every remaining transition and the final checkpoint exactly. Five native visualization frames are also exported.
3. Build an sdist, extract it to a fresh temporary directory and build the wheel there. The checkout's accumulated `build/lib` is not reused.
4. Compare every shipped wheel module/resource with current bytes under `src`, `scripts`, `data`, `scenarios` and `validation`. Reject missing, stale or duplicate files and Python caches.
5. Install the wheel offline, without dependency resolution, in a new venv outside the checkout. Scientific dependencies are reused through `system_site_packages`; this is not a fresh independent NumPy/SciPy installation.
6. Use new processes for research replay, resources, uncertainty criteria, agent interfaces and the station matrix. Station execution uses `python -I` and requires the driver, environment and exporter to import from `site-packages`.
7. Compare complete scientific station results, including checkpoints, observations, events, expected refusal details and visualization frames. Source/installed execution metadata is separate. Recheck the frozen source/resource manifest before final acceptance.

JSON receipts under `docs/benchmarks` are included in the sdist, so the explicit nominal-hour input is available after extracting the source archive. Existing receipt hashes are pinned before testing and must remain unchanged. Publishing a new receipt after acceptance does not modify those earlier inputs.

Historical tests compare every source field while acknowledging one evidenced migration: the exact v104 water cache. Only the known pair of water-file SHA256 values is allowed, with all six exact v104 cached/uncached comparisons and the applicable complete v105 nominal acceptance. Any other change, missing evidence or unsuccessful comparison is refused. This corrects obsolete guards requiring the pre-v104 hash forever; historical receipts remain evidence for their own versions and are not claimed as new executions.

This verifies matching numerical execution on one machine, not an independent physical solution. The short station matrix uses the existing declared `fast_candidate` without physics changes and has 0.10–0.15 s windows. It supplements long-duration evidence rather than replacing it.

## Reusing the full nominal hour

`--station-nominal` explicitly supplies a sealed v105 nominal receipt. Before expensive checks begin, the current numerical recipe and driver must match its source identities exactly. Acceptance requires both 3600 s source settings, all 601 windows per setting, 18 exactly repeated mathematical JSON files, the completed installed fine calculation and exact native restart with the full continuation. Partial, corrupt and stale receipts are refused.

The input is copied into the new delivery evidence. Results explicitly state `reused_existing_execution=true` and `new_hour_executed=false`: the newly built wheel is not claimed to have executed a new hour. It is freshly checked through short scenarios and exact shipped-file inspection. If the scientific recipe changed, first run the full [nominal requalification](STATION_NOMINAL_REQUALIFICATION_RU.md). A checksum protects receipt integrity, not independent authorship.

Without `--station-nominal`, the previous library-only verification remains available and reports `station_delivery.accepted=false`; it does not automatically close NPP-16.03. Neither mode grants full R1 acceptance or experimental validation.

## Outputs and failures

`progress.json` retains completed phase outcomes and log references; failed phases prevent dependent work. `tests.json` records actual pytest results. A successful sealed `report.json` includes the complete source manifest, wheel inventory and station acceptance. `station-source.json` and `station-installed.json` retain both executions; `station-nominal-input.json` is the explicit reused input. Builds and measurement logs remain under the output directory.

Station checks retain unexpected scenario failures and restart/export exceptions. Any such failure prevents acceptance even when other scenarios complete. The expected invalid-command refusal is checked separately and never reported as a physically completed episode.


## v113 acceptance

3367 library-lane tests passed in 170 source-bound resumable batches. Installed scientific station results matched exactly and all 20 gates passed. Eight short cases include seven completions and an expected refusal. A further installed-wheel repeat passed 90 HEM/delivery tests and exactly reproduced three qualification JSON files. One checkout-only import-guard test was exercised in the source environment and excluded from the installed run. The full portable run was interrupted after finding two subsequently fixed errors and is not accepted as complete. [v113 receipt](benchmarks/station_local_delivery_v113.json). NPP-16.03 is closed for this declared lane; 56/76 tasks, 20 remaining.
