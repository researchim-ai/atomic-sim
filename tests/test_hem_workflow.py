"""Stored equilibrium-water runs and declared qualification replay."""

import pytest

from atomic_model_hem import eos, workflow
from atomic_model_hem.__main__ import main
from atomic_model_workspace.common import atomic_json, read


@pytest.fixture
def output(tmp_path):
    folder = tmp_path / "demo"
    workflow.demo(folder)
    return folder


def test_saved_workflow_full_replay_and_cli(output):
    result = workflow.replay(output)
    assert result["observations"][-1]["time_s"] == 1200.0
    assert len(result["observations"][-1]["events"]) == 2
    assert result["observations"][-1]["thermodynamic_state"]["quality"] == pytest.approx(0.5)
    assert main(["replay", str(output)]) == 0


def test_saved_report_corruption_refused(output):
    result = read(output / "result.json")
    result["observations"][0]["thermodynamic_state"]["pressure_pa"] *= 1.1
    atomic_json(output / "result.json", result)
    with pytest.raises(ValueError, match="full replay"):
        workflow.replay(output)


@pytest.mark.parametrize("kind", ["source", "extra", "symlink"])
def test_invalid_output_source_inventory_or_link_refused(output, tmp_path, kind):
    if kind == "source":
        result = read(output / "result.json")
        result["source"]["if97_sha256"] = "0" * 64
        atomic_json(output / "result.json", result)
    elif kind == "extra":
        (output / "extra").write_text("x")
    else:
        (output / "input.json").rename(tmp_path / "input.json")
        (output / "input.json").symlink_to(tmp_path / "input.json")
    with pytest.raises(ValueError):
        workflow.replay(output)


def test_qualification_keeps_experiment_separate():
    r = workflow.qualify()
    assert r["inverse_grid_count"] == 25 and r["vessel_trajectories"] == 9
    assert len(r["official_table5_checks"]) == 2
    assert r["validated"] is False and r["plant_referenced"] is False
    assert all(t["exact_restart"] for c in r["vessel_cases"] for t in c["trajectories"])


def test_direct_program_with_bad_second_command_has_no_published_folder(tmp_path):
    spec = dict(
        schema_version=1,
        model=workflow.specification(),
        program=[
            dict(duration_s=10.0, command=None),
            dict(
                duration_s=10.0,
                command=dict(power_w=20000.0, expected_revision=99, controller_state={}),
            ),
        ],
    )
    with pytest.raises(ValueError):
        workflow.run(spec, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_no_overwrite(output):
    before = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        workflow.demo(output)
    assert (output / "result.json").read_bytes() == before


def test_projected_inverse_never_admitted(monkeypatch):
    state = eos.px(1e6, 0.5)
    native = eos.iapws.properties_rho_u(
        state.density_kg_m3, state.internal_energy_j_kg, 1e6, state.temperature_k, False
    )
    from dataclasses import replace

    bad = replace(native, energy_mismatch_j_kg=1.0)

    def inverse(*args, **kwargs):
        assert kwargs["allow_projection"] is False
        return bad

    monkeypatch.setattr(eos.iapws, "properties_rho_u", inverse)
    with pytest.raises(ValueError, match="Projected"):
        eos.rho_u(state.density_kg_m3, state.internal_energy_j_kg)


def test_eos_roundtrip_residual_cannot_be_hidden(monkeypatch):
    original = eos.iapws.properties_rho_u

    def changed(*args, **kwargs):
        s = original(*args, **kwargs)
        from dataclasses import replace

        return replace(s, u_j_kg=s.u_j_kg + 10.0)

    monkeypatch.setattr(eos.iapws, "properties_rho_u", changed)
    s = eos.pt(3e6, 350.0)
    with pytest.raises(ValueError, match="roundtrip"):
        eos.rho_u(
            s.density_kg_m3, s.internal_energy_j_kg, pressure_seed_pa=3e6, temperature_seed_k=350.0
        )


def test_qualification_guard_detects_damaged_pressure_model(monkeypatch):
    original = eos.rho_u

    def damaged(*args, **kwargs):
        from dataclasses import replace

        return replace(original(*args, **kwargs), pressure_pa=2e6)

    monkeypatch.setattr(eos, "rho_u", damaged)
    with pytest.raises(ValueError, match="inverse grid"):
        workflow.qualify()
