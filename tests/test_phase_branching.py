"""Topology invariance and branched flashing qualification contracts."""

from copy import deepcopy
import numpy as np
import pytest

from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from scripts.verify_phase_branching import (
    CAPS, FIELDS, MESHES, VARIANTS, canonical, differences, physical_checks, request_for,
)
from scripts.verify_phase_release import inventories


def test_all_topologies_preserve_initial_mass_energy_and_total_receiver_volume():
    stocks = []
    for cells in MESHES:
        for variant in VARIANTS:
            request = request_for(variant, cells)
            model = QualifiedHEMNetwork(request)
            stocks.append(list(inventories(model).values()))
            receivers = [n for n in request['model']['nodes'] if n['id'] != 'a']
            assert sum(n['volume_m3'] for n in receivers) == pytest.approx(.02)
            assert all(e['length_m'] == 2 and e['diameter_m'] == .05 for e in request['model']['edges'])
    np.testing.assert_allclose(stocks, np.tile(stocks[0], (len(stocks), 1)), rtol=2e-12)


@pytest.mark.parametrize('variant', ['split_equal', 'reversed', 'permuted'])
def test_equivalent_initial_physical_fields_and_stocks(variant):
    base = QualifiedHEMNetwork(request_for('parallel')).snapshot()
    other = QualifiedHEMNetwork(request_for(variant)).snapshot()
    diff = differences(canonical(base, 'parallel', 4), canonical(other, variant, 4))
    assert all(v <= FIELDS[k] for k, v in diff.items())


def test_reverse_wall_follows_same_physical_pipe_cell():
    request = request_for('reversed', 8)
    assert request['model']['walls'][0]['cell'] == 'e0/7'
    initial = request_for('parallel', 8)['model']['edges'][0]['initial']
    reversed_initial = request['model']['edges'][0]['initial']
    assert [p['pressure_pa'] for p in initial] == [p['pressure_pa'] for p in reversed_initial][::-1]


@pytest.mark.parametrize('variant,cells,cap', [('unknown',4,1e-5), ('parallel',16,1e-5), ('parallel',4,0), ('parallel',4,float('nan'))])
def test_undeclared_profile_refused(variant,cells,cap):
    with pytest.raises(ValueError): request_for(variant,cells,cap)


@pytest.mark.parametrize('damage', ['missing', 'nonfinite', 'quality', 'mass', 'energy', 'velocity'])
def test_comparison_rejects_corrupted_topology_observables(damage):
    snapshot = QualifiedHEMNetwork(request_for('parallel')).snapshot()
    first = canonical(snapshot, 'parallel', 4)
    second = deepcopy(first)
    if damage == 'missing':
        second.pop('a')
        with pytest.raises(ValueError): differences(first,second)
    elif damage == 'nonfinite':
        second['a']['pressure_pa'] = float('nan')
        with pytest.raises(ValueError): differences(first,second)
    else:
        field = {'quality':'quality','mass':'mass_kg','energy':'total_energy_j','velocity':'velocity_m_s'}[damage]
        second['a'][field] += 10 * FIELDS[field]
        assert differences(first,second)[field] > FIELDS[field]


def test_actual_asymmetric_branch_moves_mass_and_flashes():
    model = QualifiedHEMNetwork(request_for('split_unequal'))
    initial = model.snapshot()
    final = model.advance(.0002)
    assert all(physical_checks([initial,final], 'split_unequal').values())


def test_time_truncation_is_separate_from_balance_and_topology_identity():
    from scripts.verify_phase_branching import comparison_passed
    diff = {key: 0. for key in FIELDS}
    diff['total_energy_j'] = .01
    diff['mass_kg'] = 1e-8
    assert comparison_passed(diff, temporal=True)
    assert not comparison_passed(diff)
    diff['pressure_pa'] = 2 * FIELDS['pressure_pa']
    assert not comparison_passed(diff, temporal=True)


@pytest.mark.parametrize('damage', ['missing', 'nan', 'negative'])
def test_refinement_comparison_cannot_hide_bad_measurements(damage):
    from scripts.verify_phase_branching import comparison_passed
    diff = {key: 0. for key in FIELDS}
    if damage == 'missing': diff.pop('mass_kg')
    elif damage == 'nan': diff['mass_kg'] = float('nan')
    else: diff['total_energy_j'] = -1.
    with pytest.raises(ValueError): comparison_passed(diff, temporal=True)


@pytest.mark.parametrize('field', ['source', 'requests', 'program', 'caps_s'])
def test_reuse_refuses_changed_physical_protocol_before_output(tmp_path, field):
    from atomic_model_hem_qualified.model import signature
    from atomic_model_hem_qualified.qualification import CRITERIA
    from atomic_model_multiphysics.provenance import sha, seal
    from atomic_model_training.spec import atomic_json
    from scripts.verify_phase_branching import verify
    from scripts.verify_phase_release import CALLS
    reuse = tmp_path / 'reuse'
    reuse.mkdir()
    previous = dict(source=signature(), release_helper_sha256=sha('scripts/verify_phase_release.py'),
                    variants=VARIANTS, meshes=MESHES, caps_s=CAPS, program=CALLS, criteria=CRITERIA,
                    requests={v:{str(n):request_for(v,n) for n in MESHES} for v in VARIANTS})
    previous[field] = None
    atomic_json(reuse / 'protocol.json', seal(previous))
    output = tmp_path / 'refused'
    with pytest.raises(ValueError, match='Reuse source/physical protocol differs'):
        verify(output,reuse_from=reuse)
    assert not output.exists()
