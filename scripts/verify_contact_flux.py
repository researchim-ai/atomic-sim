"""Contact-preserving liquid HLLC: analytic flux, spatial and temporal evidence."""
import argparse
import math
import statistics
import time
from pathlib import Path

from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.flux import hllc
from atomic_model_hem_contact.model import signature
from atomic_model_hem_contact.reference import solve
from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_network as network
from scripts import verify_heterogeneous_network as previous

SCHEMES=('rusanov','hllc')
SPATIAL_MESHES=(2,4,8,16,32)
WAVE_CASES=('coupled_gradient','pressure_thermal_step')
WAVE_CAPS=(5e-6,2.5e-6)
STATIC_BUDGETS=dict(mass=1e-8,momentum=1e-6,energy=1e-4)
CONTACT_TEMPERATURE_BUDGET_K=1e-9
TIMES=network.TIMES


def capture(request,scheme,times=TIMES):
    model=ContactHEMNetwork(request,scheme=scheme);observations=[model.snapshot()];checkpoint=None
    for i,(a,b) in enumerate(zip(times,times[1:])):
        observations.append(model.advance(b-a))
        if i==1:checkpoint=model.checkpoint()
    replay=ContactHEMNetwork(request,scheme=scheme);replay.restore(checkpoint)
    for a,b in zip(times[2:],times[3:]):last=replay.advance(b-a)
    checks=dict(restart_exact=last==observations[-1] and replay.checkpoint()==model.checkpoint(),
        liquid_only=all(c['region']==1 and c['mass_kg']>0 for s in observations for c in s['cells']),
        mass_balance=all(abs(s['global_ledger']['mass_residual_kg'])<=1e-9 for s in observations),
        energy_balance=all(abs(s['global_ledger']['energy_residual_j'])<=1e-5 for s in observations),
        entropy=all(s['global_ledger']['numerical_entropy_residual_j_k']>=-2e-5 for s in observations),
        adiabatic=all(s['global_ledger'][k]==0. for s in observations for k in ('external_fluid_heat_j','external_wall_heat_j','external_entropy_j_k')))
    return dict(request=request,observations=observations,checkpoint=checkpoint,
                record=dict(scheme=scheme,checks=checks,passed=all(checks.values())))


def analytic_contacts():
    records=[]
    for pressure,left_t,right_t in ((1e6,430.,440.),(4e6,430.,450.),(15.5e6,550.,580.)):
        left,right=eos.pt(pressure,left_t),eos.pt(pressure,right_t)
        for velocity in (-20.,-1.,0.,1.,20.):
            for reference in (0.,7e5):
                flux=hllc(left,right,velocity,velocity,wave_bound=2500.,energy_reference=reference)
                state=left if velocity>=0 else right
                rho=state.density_kg_m3
                expected=(rho*velocity,rho*velocity**2+pressure,
                    velocity*(rho*(state.internal_energy_j_kg+.5*velocity**2)+pressure))
                actual=(float(flux[0]),float(flux[1]),float(flux[2]+reference*flux[0]))
                errors={k:abs(a-b) for k,a,b in zip(STATIC_BUDGETS,actual,expected)}
                records.append(dict(pressure_pa=pressure,velocity_m_s=velocity,energy_reference_j_kg=reference,
                    errors=errors,passed=all(errors[k]<=STATIC_BUDGETS[k] for k in errors)))
    return records


def stationary_contacts(output):
    records=[]
    for cells in SPATIAL_MESHES:
        request=previous.contact_request(cells)
        for scheme in SCHEMES:
            result=capture(request,scheme)
            first={c['id']:c for c in result['observations'][0]['cells']}
            final=result['observations'][-1]['cells']
            edges=[c for c in final if '/' in c['id']]
            l1=math.fsum(c['volume_m3']*abs(c['temperature_k']-first[c['id']]['temperature_k']) for c in edges)/math.fsum(c['volume_m3'] for c in edges)
            maximum=max(abs(c['temperature_k']-first[c['id']]['temperature_k']) for c in final)
            record=dict(cells=cells,scheme=scheme,l1_temperature_k=l1,maximum_temperature_change_k=maximum,
                checks=result['record']['checks'],passed=result['record']['passed'] and (scheme!='hllc' or maximum<=CONTACT_TEMPERATURE_BUDGET_K))
            records.append(record);atomic_json(output/f'contact-{cells}-{scheme}.json',seal(dict(record=record,native=result)))
            print('contact',cells,scheme,record['passed'],l1,flush=True)
    return records


def performance():
    """Measured tradeoff, not a speed acceptance gate or dedicated CPU claim."""
    records=[]
    for case in ('thermal_step','pressure_thermal_step'):
        runs=[]
        for trial in range(3):
            order=SCHEMES if trial%2==0 else tuple(reversed(SCHEMES))
            for scheme in order:
                started=time.process_time()
                model=ContactHEMNetwork(network.request_for(case,4,1e-5),scheme=scheme)
                for a,b in zip(TIMES,TIMES[1:]):model.advance(b-a)
                checkpoint=model.checkpoint()
                runs.append(dict(trial=trial,scheme=scheme,cpu_s=time.process_time()-started,
                                 checkpoint_sha256=checkpoint['sha256']))
        medians={scheme:statistics.median(r['cpu_s'] for r in runs if r['scheme']==scheme) for scheme in SCHEMES}
        records.append(dict(case=case,cells=4,cap_s=1e-5,runs=runs,median_cpu_s=medians,
            hllc_to_rusanov_cpu_ratio=medians['hllc']/medians['rusanov'],
            scope='Constructor, four advances, checkpoint;3 paired alternating runs; no speed qualification'))
    return records


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,network.__file__,previous.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,schemes=SCHEMES,
        static_budgets=STATIC_BUDGETS,contact_temperature_budget_k=CONTACT_TEMPERATURE_BUDGET_K,
        wave_budgets=previous.REFERENCE_BUDGETS,wave_cases=WAVE_CASES,wave_meshes=(2,4),wave_caps_s=WAVE_CAPS,
        spatial_meshes=SPATIAL_MESHES,validated=False,plant_referenced=False,
        scope='Opt-in single-phase adiabatic HLLC; not two-phase/CHF or independent external CFD validation')))
    static=analytic_contacts();contact=stationary_contacts(output);waves=[]
    for case in WAVE_CASES:
        for cells in (2,4):
            request=network.request_for(case,cells,1e-5)
            references=[solve(request,TIMES,method) for method in ('DOP853','Radau')]
            natives={}
            for cap in WAVE_CAPS:
                request=network.request_for(case,cells,1e-5);request['model']['max_step_s']=cap
                native=capture(request,'hllc');natives[str(cap)]=native
                for reference in references:
                    result=network.compare(native,previous.as_peer(reference['observations']))
                    result['checks']['strict_agreement']=all(result['maximum_errors'][k]<=v for k,v in previous.REFERENCE_BUDGETS.items())
                    result['passed']=all(result['checks'].values())
                    record=dict(case=case,cells=cells,cap_s=cap,method=reference['method'],**result)
                    waves.append(record);print('wave',case,cells,cap,reference['method'],record['passed'],flush=True)
            atomic_json(output/f'wave-{case}-{cells}.json',seal(dict(references=references,natives=natives)))
    # Old family remains executable and has identical dynamics in baseline mode.
    request=network.request_for('coupled_gradient',2,1e-5)
    legacy=QualifiedHEMNetwork(request);baseline=ContactHEMNetwork(request,scheme='rusanov')
    legacy_equal=legacy.advance(.00025)==baseline.advance(.00025)
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items())
    timings=performance()
    unchanged=unchanged and source==signature() and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,source_unchanged=unchanged,legacy_rusanov_observations_identical=legacy_equal,
        passed=unchanged and legacy_equal and all(r['passed'] for r in static+contact+waves),
        analytic_contacts=static,stationary_contacts=contact,waves=waves,performance=timings,
        validated=False,plant_referenced=False,external_modelica_hllc_compared=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    return 0 if run(parser.parse_args().output)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
