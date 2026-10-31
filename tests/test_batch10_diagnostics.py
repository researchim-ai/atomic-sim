import pytest

from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.initialization import equilibrated_config
from scripts.generate_pipe_pulse import make_spec
from scripts.verify_pipe_mesh import configuration
from scripts.verify_pipe_transient import ACCEPTANCE_CRITERIA, _pipe_signature, assess


def test_solver_counters_include_default_numerical_jacobian_probes(monkeypatch):
    m = DynamicWallChannel(equilibrated_config(configuration(3, 0.0)))
    calls = 0
    original = m.derivatives

    def track(t, y):
        nonlocal calls
        calls += 1
        return original(t, y)

    monkeypatch.setattr(m, "derivatives", track)
    m.enable_continuation(0.01)
    m.advance(0.01, method="BDF")
    counters = m.solver_diagnostics()
    assert counters["accepted_rhs_evaluations"] == calls
    assert calls > m._continuation.solver.nfev
    assert counters["failed_rhs_evaluations"] == 0


def test_relabelled_doubling_grids_and_nonacceptance_norms():
    rows = []
    for n in (4, 8, 16, 16):
        rows.append(
            {
                t: dict(
                    right_cell_temperature_k=500.0 + s / n,
                    mean_wall_temperature_k=505.0 + s / n,
                    midpoint_pressure_pa=7e6,
                    inlet_mass_flow_kg_s=1.0,
                    outlet_mass_flow_kg_s=1.0,
                )
                for t, s in ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0))
            }
        )
    metrics, _, checks = assess(rows, grids=(8, 16, 32))
    assert all(checks.values())
    m = metrics["right_cell_temperature_k"]
    assert m["observed_order"] == pytest.approx(1.0)
    assert "difference_8_16_k" in m and "difference_16_32_k" in m
    diag = m["additional_diagnostics_not_used_for_acceptance"][0]
    assert diag["l1_k_s"] == pytest.approx(0.125)
    assert diag["max_difference_time_s"] == 1.0
    rows[-1][1.0]["midpoint_pressure_pa"] += 2.0
    assert not assess(rows)[2]["temporal_pressure"]
    assert ACCEPTANCE_CRITERIA["maximum_pressure_temporal_pa"] == 1.0


def test_inconsistent_trace_times_are_refused_before_arithmetic():
    with pytest.raises(ValueError):
        assess([{}, {}, {}, {}])
    row = {0.0: {}, 1.0: {}}
    with pytest.raises(ValueError):
        assess([row, row, row, {0.0: {}}])


def test_schemes_are_part_of_comparison_identity():
    donor = _pipe_signature(make_spec(4))
    muscl = _pipe_signature(make_spec(4, scheme="muscl_mc"))
    assert donor["enthalpy_transport"] == "donor"
    assert muscl["enthalpy_transport"] == "muscl_mc"
    assert donor != muscl


def test_generator_cli_reports_bad_scheme_geometry_without_creating_output(tmp_path):
    from scripts.generate_pipe_pulse import main

    target = tmp_path / "input.json"
    with pytest.raises(SystemExit) as exc:
        main(["--cells", "2", "--scheme", "muscl_mc", "--output", str(target)])
    assert exc.value.code == 2 and not target.exists()


def test_generator_cli_does_not_overwrite_existing_input(tmp_path):
    from scripts.generate_pipe_pulse import main

    target = tmp_path / "input.json"
    target.write_text("keep")
    with pytest.raises(SystemExit) as exc:
        main(["--cells", "4", "--output", str(target)])
    assert exc.value.code == 2 and target.read_text() == "keep"
