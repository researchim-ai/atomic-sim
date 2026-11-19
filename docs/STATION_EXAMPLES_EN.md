# Five first programs with the station environment

[Русский](STATION_EXAMPLES_RU.md)

`atomic_model_station_examples` ships in the wheel. Once the library and its scientific dependencies are installed, the examples run from any working directory without a source checkout.

Each example saves actions, observations, a final checkpoint and a report. Use a new output directory for each run. Existing result directories are rejected to preserve earlier experiments.

## First calculation

```bash
python -m atomic_model_station_examples first --output results/first
```

The environment uses a fixed seed and advances one 0.02 s step. `trajectory.json` records the initial instrument readings and the transition: new readings, reward, termination flags and step information. `report.json` checks completion of the declared step. This is a first calculation from the initial state; it does not assess steady operation.

## Load commands

```bash
python -m atomic_model_station_examples load --output results/load
```

Three steps request 100 → 99 → 100% load. Checks confirm that the ordinary control dispatcher selects each request. Control is manual and other settings are held. The 0.06 s horizon demonstrates command and measurement flow. Settling and tracking performance require the separate, longer load qualification scenario.

## Equipment failure

```bash
python -m atomic_model_station_examples fault --output results/fault
```

At t=0.02 s, a scheduled loss of the feed-pump drive takes effect. The operator continues requesting an enabled pump. Checks verify the failure epoch, active fault and the distinction between requested and effective drive state. The physical model retains fluid momentum rather than zeroing the flow on command.

Fault truth is inspected by the evaluator after execution. It remains in the privileged research checkpoint and is not added to instrument readings.

## A simple agent

```bash
python -m atomic_model_station_examples agent --output results/agent
```

The agent receives instrument observations and the public action schema. It obtains a power target from a valid initial measurement, enables the existing rod control loop and supplies that target. Other loops remain manual. The baseline policy also returns an alarm flag based on availability and agreement of two pressure measurements.

The policy receives no environment object, checkpoint, latent physical state or fault schedule. The example refuses to proceed without a valid initial power measurement. This short episode demonstrates the interface; it does not evaluate success on an industrial control task.

## Replay

```bash
python -m atomic_model_station_examples replay --output results/replay
```

After the first step, `midpoint.json` is saved. A new environment is reconstructed through the ordinary `StationEnv.restore` API and executes the remaining actions. Every suffix transition and the final checkpoint must exactly match the uninterrupted run. Restore replays command history and therefore requires host compute time.

## Reading the results

`example_completed` means that all declared steps ran. `physical_scenario_qualified`, `industrial_qualification` and `task_success_evaluated` remain false. These short programs introduce the API. Checkpoints contain privileged model state; use measured environment observations when training a policy.

The same examples are available from Python:

```python
from atomic_model_station_examples import run_example

report = run_example("replay", "results/python-replay")
assert report["passed"]
```

For custom policies, use `StationEnv.reset`, `StationEnv.available_actions` and `StationEnv.step`. The separate `advance_hybrid_actions` API adds atomic heat auditing of a declared action sequence. A full nominal hour uses `python -m scripts.requalify_station_nominal`; it is independent of these introductory runs.

## Delivery acceptance v106

All five examples ran from source and from an installed wheel outside the checkout. Nine tests passed and were repeated from the wheel. All 16 result JSON documents matched exactly, with no fields excluded. NPP-16.02 is accepted for these short API tutorials; 49/76 plan items are complete and 27 remain open. Long physical scenarios and the full R1 release require separate acceptance.

[Machine-readable receipt](benchmarks/station_examples_v106.json).
