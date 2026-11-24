"""Aggregate acceptance of exact caches, sparse LU and the qualified PWR path."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_training.spec import atomic_json,read_json
from scripts import qualify_exchanger_cache as operator
from scripts import qualify_sparse_steady as steady
from scripts import qualify_pwr_sparse as pwr
from scripts.qualify_pwr_distributed_fuel import checks,relative
from coupling.pwr_sparse import SparseLocalModeratorPWRSession
from coupling.pwr_moderator import LocalModeratorPWRSession
from coupling.refinement import coupled_pwr_mesh

EXPECTED = {'operator': 'e8fc337c38affbc95bd3afbc69443e92d3a8d11b5e78d4cbedf793736c6d1dd5', 'steady': 'd0b945776c2f98675fb0678ab4b966f0089d55e856f62e422c2586d605ca1160', 'pwr': '3718b59c6ec0e994258baa9ee0b6796d14af36639c93a4e840ecf715745197b4'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source():
    return dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_qualification_digests={name:digest(module.source()) for name,module in
            (('operator',operator),('steady',steady),('pwr',pwr))})


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    frozen=source()
    if frozen['dependency_qualification_digests']!=EXPECTED:raise ValueError('Qualified dependency source changed; repeat its full matrix before aggregation')
    out.mkdir(parents=True)
    operator.run(out/'operator');steady.run(out/'steady')
    science={name:unseal(read_json(out/name/'scientific.json',limit=16*1024*1024)) for name in ('operator','steady')}
    pwr_rows={};checkpoint=None
    for name,cls in (('dense',LocalModeratorPWRSession),('sparse',SparseLocalModeratorPWRSession)):
        s=cls(**coupled_pwr_mesh(2),max_step_s=.05,rtol=2e-9)
        steps=[dict(duration_s=.1,patch={'rod_target':s.controls['rod_target']+.01}),dict(duration_s=.4,patch=None)]
        rows=[s.advance(**steps[0])];saved=json.loads(json.dumps(s.checkpoint(),allow_nan=False));rows.append(s.advance(**steps[1]))
        if name=='sparse':
            restored=cls.restore(saved)
            checkpoint=restored.advance(**steps[1])==rows[-1] and restored.checkpoint()==s.checkpoint()
        pwr_rows[name]=rows
    errors=[relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(pwr_rows['dense'],pwr_rows['sparse'])]
    gates=dict(dependency_sources=frozen['dependency_qualification_digests']==EXPECTED,
        operator=science['operator']['report']['passed'],steady=science['steady']['report']['passed'],
        pwr_budgets=all(all(checks(r).values()) for rows in pwr_rows.values() for r in rows),
        pwr_dense_comparison=max(errors)<=1e-5,pwr_restart=checkpoint is True,
        pwr_csc=all(r['sparse_jacobian']['all_nonzero_coefficients_retained'] is True and r['sparse_jacobian']['drop_tolerance'] is None for r in pwr_rows['sparse']))
    report=dict(passed=all(gates.values()) and source()==frozen,source=frozen,gates=gates,
        maximum_pwr_backend_relative=max(errors),prior_full_pwr_matrix_source_locked=True,
        prior_pwr_scope='v146 insertion/return/four-second SCRAM on 2/2/4, twelve trajectories and six restarts; current source digest must match',
        current_pwr_scope='Additional two matched half-second trajectories and one exact sparse restart',
        closures_scope='Exact-key transition cache; small linear sparse LU reuse; admitted full-Jacobian CSC Radau path',
        not_claimed=['Structural Jacobian assembly without dense probes','Universal acceleration','Long-horizon station acceptance','Experimental validation'],
        closed_tasks=[],experimental_validation=False,full_library_profile_executed=False)
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,components=science,pwr=pwr_rows)))
    if not report['passed']:raise ValueError('Numerical reuse aggregate failed; attempts retained')
    return dict(passed=True,gates=gates)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);print(run(p.parse_args().output))
