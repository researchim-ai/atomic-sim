from copy import deepcopy
import json
from pathlib import Path
import pytest
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json,read_json
from scripts import verify_local_tests as batch


@pytest.fixture
def repo(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'tests').mkdir(parents=True)
    (root/'pyproject.toml').write_text('[tool.pytest.ini_options]\naddopts=""\n')
    (root/'tests/test_a.py').write_text('def test_one(): assert 1+1==2\ndef test_two(): assert True\n')
    (root/'tests/test_b.py').write_text('def test_three(): assert True\n')
    (root/'docs/benchmarks').mkdir(parents=True)
    (root/'docs/benchmarks/input.json').write_text('{}')
    monkeypatch.setattr(batch,'ROOT',root)
    monkeypatch.setattr(batch.local,'selection',lambda suite:['tests'])
    return root,tmp_path/'results'


def test_real_pytest_batches_cover_all_tests_and_resume_without_reexecution(repo,monkeypatch):
    monkeypatch.setenv('PYTEST_ADDOPTS','-k no_such_test')
    monkeypatch.setenv('PYTEST_PLUGINS','no_such_plugin')
    root,out=repo;p=batch.prepare(out,batch_files=1)
    assert len(p['nodeids'])==3 and len(p['batches'])==2
    first=batch.run_batch(out,0);assert first['pytest']['counts']['passed']==2
    assert batch.run_batch(out,0)==first
    assert len(list((out/'batch-000').glob('attempt-*')))==1
    with pytest.raises(OSError):batch.accept_batches(out,suite='library')
    batch.run_batch(out,1);r=batch.accept_batches(out,suite='library')
    assert r['counts']==dict(passed=3,failed=0,skipped=0,errors=0,collected=3)
    assert r['reused_completed_test_executions'] and len(r['batch_receipts_sha256'])==2
    with pytest.raises(ValueError):batch.accept_batches(out,suite='portable')


def test_partial_attempt_cannot_count_as_pass_and_is_retained(repo):
    root,out=repo;batch.prepare(out,batch_files=1)
    interrupted=out/'batch-000'/('attempt-'+'0'*32);interrupted.mkdir(parents=True)
    (interrupted/'pytest.log').write_text('interrupted before receipt')
    result=batch.run_batch(out,0)
    assert result['attempt']!=interrupted.name and (interrupted/'pytest.log').read_text()=='interrupted before receipt'
    assert len(list((out/'batch-000').glob('attempt-*')))==2

@pytest.mark.parametrize('change',['code','test','evidence','profile','runtime'])
def test_epoch_or_selection_changes_refused_before_tests(repo,change,monkeypatch):
    root,out=repo;batch.prepare(out,batch_files=1)
    if change=='code':(root/'src').mkdir();(root/'src/new.py').write_text('x=1')
    if change=='test':(root/'tests/test_a.py').write_text('def test_replaced(): pass')
    if change=='evidence':(root/'docs/benchmarks/input.json').write_text('{"changed":true}')
    if change=='profile':monkeypatch.setattr(batch.local,'selection',lambda suite:['different'])
    if change=='runtime':
        original=batch.snapshot
        monkeypatch.setattr(batch,'snapshot',lambda:{**original(),'runtime':dict(python='different')})
    with pytest.raises(ValueError,match='epoch'):batch.run_batch(out,0)
    assert not (out/'batch-000').exists()


def test_new_evidence_receipt_does_not_modify_existing_validation_inputs(repo):
    root,out=repo;batch.prepare(out,batch_files=1)
    (root/'docs/benchmarks/new.json').write_text('{}')
    assert batch.run_batch(out,0)['pytest']['counts']['passed']==2

@pytest.mark.parametrize('name',['pytest.log','nodeids.json','pytest.json'])
def test_changed_batch_artifacts_cannot_be_reused(repo,name):
    root,out=repo;batch.prepare(out,batch_files=1);r=batch.run_batch(out,0)
    artifact=out/'batch-000'/r['attempt']/name;artifact.write_bytes(artifact.read_bytes()+b' ')
    with pytest.raises(ValueError,match='artifact changed'):batch.run_batch(out,0)


def test_forged_counts_fail_even_if_hashes_and_envelope_are_recomputed(repo):
    root,out=repo;batch.prepare(out,batch_files=1);r=batch.run_batch(out,0)
    r['pytest']['counts']['passed']=1
    work=out/'batch-000'/r['attempt'];atomic_json(work/'pytest.json',r['pytest'])
    r['files_sha256']['pytest.json']=sha(work/'pytest.json');atomic_json(out/'batch-000/receipt.json',seal(r))
    with pytest.raises(ValueError,match='Every required test'):batch.run_batch(out,0)

@pytest.mark.parametrize('defect',['overlap','missing'])
def test_resealed_plan_with_incomplete_or_overlapping_coverage_is_refused(repo,defect):
    root,out=repo;batch.prepare(out,batch_files=1);p=unseal(read_json(out/'plan.json'))
    if defect=='overlap':p['batches'][1].append(p['batches'][0][0])
    else:p['batches'][0].pop()
    atomic_json(out/'plan.json',seal(p))
    with pytest.raises(ValueError,match='coverage'):batch.run_batch(out,0)


def test_failed_test_keeps_attempt_and_never_creates_accepted_receipt(repo):
    root,out=repo;(root/'tests/test_b.py').write_text('def test_three(): assert False')
    batch.prepare(out,batch_files=1)
    with pytest.raises(ValueError,match='failed or interrupted'):batch.run_batch(out,1)
    assert not (out/'batch-001/receipt.json').exists()
    logs=list((out/'batch-001').glob('attempt-*/pytest.log'));assert len(logs)==1
    assert 'FAILED' in logs[0].read_text()
    with pytest.raises(OSError):batch.accept_batches(out,suite='library')


def test_collection_errors_and_empty_selection_cannot_create_plan(repo):
    root,out=repo;(root/'tests/test_a.py').write_text('def broken(')
    with pytest.raises(ValueError,match='collection failed'):batch.prepare(out)
    assert not (out/'plan.json').exists() and (out/'collection.log').exists()


def test_output_and_batch_budgets_are_guarded(repo,tmp_path):
    root,out=repo
    with pytest.raises(ValueError):batch.prepare(root/'tests/results')
    with pytest.raises(ValueError):batch.prepare(out,batch_files=True)
    batch.prepare(out,batch_files=1)
    for index in (-1,2,True):
        with pytest.raises(ValueError):batch.run_batch(out,index)
    with pytest.raises(ValueError):batch.prepare(out)
    alias=tmp_path/'alias';alias.symlink_to(out,target_is_directory=True)
    with pytest.raises(ValueError):batch.run_batch(alias,0)

@pytest.mark.parametrize('counts',[
 dict(passed=True,failed=0,skipped=0,errors=0,collected=1),
 dict(passed=0,failed=0,skipped=1,errors=0,collected=1),
 dict(passed=1,failed=0,skipped=0,errors=1,collected=1),
])
def test_boolean_counts_skipped_library_tests_and_errors_do_not_pass(counts):
    with pytest.raises(ValueError):batch._counts(dict(schema_version=1,counts=counts,exit_code=0,gate_failures=[]),['test'],suite='library')


def test_unknown_receipt_schema_cannot_be_silently_reused():
    c=dict(passed=1,failed=0,skipped=0,errors=0,collected=1)
    with pytest.raises(ValueError):batch._counts(dict(schema_version=2,counts=c,exit_code=0,gate_failures=[]),['test'],suite='library')


def test_successful_retry_cannot_hide_confirmed_earlier_test_failure(repo):
    root,out=repo;batch.prepare(out,batch_files=1)
    batch.run_batch(out,0);batch.run_batch(out,1)
    previous=out/'batch-000'/('attempt-'+'f'*32);previous.mkdir()
    atomic_json(previous/'pytest.json',dict(counts=dict(failed=1,errors=0)))
    with pytest.raises(ValueError,match='cannot be hidden'):batch.accept_batches(out,suite='library')
