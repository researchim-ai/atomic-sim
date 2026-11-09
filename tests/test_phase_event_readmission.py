"""Synthetic admission fixtures below are not physical event evidence."""
from copy import deepcopy
from pathlib import Path
import pytest
from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import compare_phase_event_campaign as bench


def synthetic(directory):
    directory.mkdir()
    modules=(bench.event,bench.phase_wall_case,bench.verify_phase_cycles)
    protocol=dict(source=signature(),drivers={m.__file__:sha(m.__file__) for m in modules})
    refs=[dict(case=c,quadrature_time_s=.005,quadrature_error_s=0.,passed=True,
               ode=[dict(method=m,time_s=.005) for m in ('DOP853','Radau')]) for c in bench.event.CASES]
    native=[dict(case=c,cells=n,cap_s=cap,lower_s=.005,upper_s=.005,
                 signed_energy_bracket_j_kg=[0.,0.],maximum_event_error_s=0.,passed=True,restart_exact=True)
            for c in bench.event.CASES for n in bench.event.MESHES for cap in bench.event.CAPS]
    report=dict(completed=True,source_unchanged=True,native_passed=True,native=native,references=refs)
    return protocol,report


def save(directory,protocol,report):
    atomic_json(directory/'protocol.json',seal(protocol));atomic_json(directory/'report.json',seal(report))


def test_exact_source_reference_can_be_admitted_without_recomputation(tmp_path):
    directory=tmp_path/'refs';p,r=synthetic(directory);save(directory,p,r)
    assert bench.read_reference(directory)==r


@pytest.mark.parametrize('damage',['source','driver','prefix','duplicate','nan_root','wide','false_error','nan_ref','bad_ode','restart','failed'])
def test_resealed_incomplete_or_nonfinite_reference_refused(tmp_path,damage):
    directory=tmp_path/'refs';p,r=synthetic(directory)
    if damage=='source':p['source']={}
    elif damage=='driver':p['drivers'][bench.event.__file__]='wrong'
    elif damage=='prefix':r['native'].pop()
    elif damage=='duplicate':r['native'][-1]=deepcopy(r['native'][0])
    elif damage=='nan_root':r['native'][0]['lower_s']=float('nan')
    elif damage=='wide':r['native'][0]['upper_s']+=1e-6
    elif damage=='false_error':r['native'][0]['maximum_event_error_s']=1e-9
    elif damage=='nan_ref':r['references'][0]['quadrature_time_s']=float('nan')
    elif damage=='bad_ode':r['references'][0]['ode'][0]['time_s']+=1e-5
    elif damage=='restart':r['native'][0]['restart_exact']=False
    else:r['native_passed']=False
    with pytest.raises(ValueError):
        # The canonical envelope itself rejects NaN; finite resealed damage
        # reaches the numerical reference-admission checks.
        save(directory,p,r)
        bench.read_reference(directory)


@pytest.mark.parametrize('campaigns',[[],['a','a'],['parent/a','other/a']])
def test_empty_or_ambiguous_campaign_selection_refused_before_output(tmp_path,campaigns):
    with pytest.raises(ValueError):bench.run(tmp_path/'out',reference_from=tmp_path/'missing',campaigns=campaigns)
    assert not (tmp_path/'out').exists()
