# Station viewer: flows, equipment and measurements

[Русский](STATION_VIEW_RU.md). `/ru/station/` and `/en/station/` display native StationEnv frames and are linked from the homepage. The viewer works locally or on static hosting. Python calculates trajectories; the browser selects recorded frames. There is no connection to a running plant, remote control or JavaScript physics solver.

## Start locally

With the library installed, run from the repository root:

```bash
python -m atomic_model_station_view --output artifacts/my-station-recording.json
cd web
npm run dev
```

Visit `http://127.0.0.1:4173/en/station/`. The bundled recording has five frames: initial state, feed-pump off/on, a DHR request and SCRAM. A pressure instrument failure is scheduled on ticks 1–3, using the native half-open interval. The tick is 0.05 s and the entire window is 0.20 s. Instrument-triggered protection stays latched after signal recovery. This is an interface example, not long-transient qualification.

To export your own native trajectory:

```python
from atomic_model_station_env import EnvConfig, StationEnv
from atomic_model_station_view import capture, document, write

env = StationEnv(EnvConfig(max_steps=2, tick_s=0.05, automatic=False))
env.reset()
frames = [capture(env)]
env.step({"operator": {"feed_pump_on": False}})
frames.append(capture(env))
env.step({"operator": {"feed_pump_on": True}})
frames.append(capture(env))
write("station-recording.json", document(frames))
```

Choose “Open your calculation”. The file remains in the browser. Limits are 8 MiB and 1025 strictly ordered frames. Export refuses existing output files; choose a fresh name. The exporter also works from an installed wheel. `capture` snapshots a copy of the physical owner without changing the checkpoint or advancing instruments.

## Reading the display

The graph groups the reactor and steam generator into one node. It is an aggregate research-model topology, not manufacturer piping. Nine edges carry steam, feedwater, condensate, condenser heat, external heat rejection, gross generation, auxiliary consumption, grid exchange and DHR heat. Mass flows use kg/s; power is stored in W and displayed in MW. Negative grid exchange remains import and reverses the displayed direction. Line widths are not quantitative. Heat and mass flows are not added together.

Equipment distinguishes operator requests, actual drive power, rotational speed and check-valve state. An unpowered drive does not imply an immediate loss of rotation or flow. Native backend values are preserved: for example, the feed check valve has already closed by the end of the first demonstration tick.

Operator measurements remain separate from physical truth: values, units, quality codes, age, sampling and delivery times. Untrusted readings stay marked; physical pressure is never substituted into the instrument table. Protection reasons are translated while retaining their machine codes, and event parameters are preserved. Backend SCRAM and the supervisory protection latch are separate flags.

Balances show accumulated relative energy, mass and boron residuals, signed energy residual in J and instantaneous defect in W. Small residuals support numerical accounting, not experimental validation. Differences between input and output power may change stored energy; instantaneous power ratios are not presented as steady efficiency.

## File contract and trust

`station-view-file/v1` contains the exact UTF-8 document string and its SHA256; the inner format is `station-view/v1`. Both loaders check schema, units, clocks, equipment and instrument records. The browser verifies SHA256 before replacing the current recording. A rejected file leaves the previous recording intact. Integrity is not authorship: a file author can recompute the hash. Station/exporter source hashes identify claimed provenance; displaying an older file does not grant it current physical validation.

Exports explicitly set `privileged=true`. They contain physical values, protection reasons and events, so must not replace the measured observations given to an agent. Imported strings are rendered as DOM text, never executable HTML. Files are not uploaded; no storage or analytics is used. Web Crypto requires a supporting browser on localhost or HTTPS.

## v112 acceptance

68 targeted Python tests and 12 website tests passed. All 17 new Python tests were repeated from the installed wheel outside the checkout, and the entire demonstration JSON matched. Browser checks covered both languages, final-frame selection, rejected import retaining the current recording and mobile layout. Export tests also verify an unchanged subsequent trajectory and native primary-pump rotation after power loss.

The [v112 receipt](benchmarks/station_view_v112.json) accepts NPP-16.01 for local recording inspection of this configuration. Long transients, physical validation, full R1 release and online supervisory operation remain separate. The scientific station backend is unchanged.
