"""Opt-in local HLLC: nonlinear wave coverage, temporal graph and restart gates."""
import argparse
from pathlib import Path
from atomic_model_hem_contact.flux import local_hllc,local_envelope
from atomic_model_hem_contact.model import signature
from atomic_model_hem_contact.reference import solve
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_liquid_riemann as physical
from scripts import liquid_riemann as oracle
from scripts import benchmark_modelica_hllc as cases
from scripts import benchmark_modelica_network as network
from scripts import verify_contact_flux as original
from scripts import verify_heterogeneous_network as previous

SCHEME='hllc-local'
CAPS=(5e-6,2.5e-6)
LONG_TIMES=(0.,.0025,.005,.0075,.01)


def face_cases():
    rows=[]
    for case in physical.cases():
        name,left,right,vl,vr,bracket=case
        assessed=physical.evaluate(case)
        solution=oracle.solve(left,right,vl,vr,pressure_bracket=bracket)
        bound=local_envelope(left,right,vl,vr,maximum=2500.)
        exact=solution.interface_flux()
        flux=local_hllc(left,right,vl,vr,wave_bound=2500.,energy_reference=0.)
        relative=None if abs(exact[0])<1e-12 else abs(float(flux[0])-exact[0])/abs(exact[0])
        checks=dict(physical_oracle=assessed['passed'],
            exact_wave_coverage=max(abs(solution.left_outer_speed_m_s),abs(solution.right_outer_speed_m_s))<=bound,
            diagnostic_reproduced=list(map(float,flux))==assessed['candidates'][1]['flux'])
        # No universal all-case accuracy claim: target the pressure-driven waves.
        if name in ('moderate','wide','reversed') or name.startswith('acoustic-'):
            checks['mass_flux_relative_error_below_3_percent']=relative<.03
            checks['improves_fixed_mass_flux_error']=relative<assessed['candidates'][0]['relative_mass_flux_error']
        rows.append(dict(case=name,wave_bound_m_s=bound,relative_mass_flux_error=relative,
                         checks=checks,passed=all(checks.values())))
    return rows


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,physical.__file__,oracle.__file__,original.__file__,cases.__file__,network.__file__,previous.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,scheme=SCHEME,caps_s=CAPS,
        times_s=original.TIMES,long_times_s=LONG_TIMES,temporal_budgets=previous.REFERENCE_BUDGETS,
        signal_margin=1.01,global_cfl_unchanged=True,validated=False,plant_referenced=False,
        scope='Bounded liquid adiabatic graph; nonlinear wave coverage checked in 15 cases, not globally proved')))
    faces=face_cases();waves=[];long=[];contacts=[]
    for case in ('coupled_gradient','pressure_thermal_step'):
        for cells in (2,4):
            request=cases.request_for(case,cells,CAPS[0])
            references=[solve(request,original.TIMES,method,scheme=SCHEME) for method in ('DOP853','Radau')]
            natives={}
            for cap in CAPS:
                request=cases.request_for(case,cells,cap)
                native=original.capture(request,SCHEME);natives[str(cap)]=native
                for reference in references:
                    assessment=network.compare(native,previous.as_peer(reference['observations']))
                    assessment['checks']['strict_agreement']=all(assessment['maximum_errors'][k]<=v for k,v in previous.REFERENCE_BUDGETS.items())
                    assessment['passed']=all(assessment['checks'].values())
                    row=dict(case=case,cells=cells,cap_s=cap,method=reference['method'],**assessment)
                    waves.append(row);print('temporal',case,cells,cap,reference['method'],row['passed'],flush=True)
            atomic_json(output/f'wave-{case}-{cells}.json',seal(dict(references=references,natives=natives)))
    for cells in (2,4,8,16,32):
        result=original.capture(previous.contact_request(cells),SCHEME)
        first=result['observations'][0]['cells'];last=result['observations'][-1]['cells']
        error=max(abs(a['temperature_k']-b['temperature_k']) for a,b in zip(first,last,strict=True))
        row=dict(cells=cells,maximum_temperature_change_k=error,checks=result['record']['checks'],
                 passed=result['record']['passed'] and error<=original.CONTACT_TEMPERATURE_BUDGET_K)
        contacts.append(row);atomic_json(output/f'contact-{cells}.json',seal(dict(record=row,native=result)))
    for case in ('pressure_thermal_step','reverse_pressure_step','wide_pressure_step'):
        request=cases.request_for(case,2,CAPS[0])
        result=original.capture(request,SCHEME,times=LONG_TIMES)
        row=dict(case=case,cells=2,horizon_s=LONG_TIMES[-1],**result['record'])
        long.append(row);atomic_json(output/f'long-{case}.json',seal(result))
        print('long',case,row['passed'],flush=True)
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,passed=unchanged and all(r['passed'] for r in faces+waves+contacts+long),
        source_unchanged=unchanged,face_cases=faces,waves=waves,stationary_contacts=contacts,long_runs=long,
        validated=False,plant_referenced=False,external_modelica_local_hllc_compared=False,
        whole_plant_compared=False,global_wave_bound_proved=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    return 0 if run(parser.parse_args().output)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
