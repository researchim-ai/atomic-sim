"""Separate execution, numerical-resolution and statistical acceptance gates."""
import argparse
import json
from pathlib import Path
import numpy as np
from atomic_model_station_spectral import WindowPolicy,analyse_window
from atomic_model_station_spectral.data import load_fixture
from atomic_model_station_spectral.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read


def identity():return dict(model=model_identity(),driver_sha256=sha(__file__))


def protocol():
    return dict(source=identity(),criteria=dict(native_audit_absolute=1e-9,mean_absolute=5e-11,wall_absolute=1e-9,
        divergence_l2=1e-7,terminal_amplitude_relative_to_nonzero_peak=1e-3,upper_band_energy_fraction=1e-5,
        restart_rms=1e-10,time_rms=1e-5,space_rms=1e-3),
        statistics_policy=WindowPolicy().__dict__,
        tail_norm='Integrated half-plane energy; terminal amplitude divided by largest nonzero-mode shell, not by the mean',
        thermal_trace='Native wall-gradient diagnostic is a magnitude; no signed Nu is inferred',
        execution_gate='Complete real-mode experiments and agreement of independent/native audit. Physical and statistical gates separate.',
        scope='Short3D integration probes at Re5300,Pr0.7,C0; no statistical DNS or plant qualification')


def run(output,reference=None):
    out=Path(output).absolute();p=protocol();source=p['source'];limits=p['criteria']
    if out.exists() or any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if reference and read(Path(reference)/'protocol.json')!=p:raise ValueError('Protocol identity differs')
    fixture=load_fixture();out.mkdir(parents=True);atomic_json(out/'protocol.json',seal(p));audits={};passed=True
    for name,row in fixture['records'].items():
        differences=[];checks=[]
        for s,native in zip(row['snapshots'],row['native_audits']):
            error=max(abs(s['perturbation_kinetic_integral']-native['kinetic_integral']),
                      abs(s['temperature_perturbation_energy_integral']-native['thermal_integral']),abs(s['divergence_l2']-native['divergence_l2']))
            differences.append(error)
            checks.append(dict(time=s['time'],spatial_modes_present=s['spatial_modes_present'],
                means_passed=abs(s['area_mean_velocity']-.5)<limits['mean_absolute'] and abs(s['area_mean_temperature']-.5)<limits['mean_absolute'],
                walls_passed=max(s['no_slip_mode_maximum'],s['temperature_wall_mode_maximum'])<limits['wall_absolute'],
                continuity_passed=s['divergence_l2']<limits['divergence_l2'],
                tails_passed=all(v['terminal_amplitude_ratio'] is not None and v['terminal_amplitude_ratio']<limits['terminal_amplitude_relative_to_nonzero_peak'] and v['upper_band_energy_fraction']<limits['upper_band_energy_fraction'] for v in (s['axial_tail'],s['azimuthal_tail']))))
        native_ok=bool(len(differences)==3 and max(differences)<limits['native_audit_absolute'])
        epochs=row['spec']['start']+np.arange(row['spec']['steps'])*row['spec']['step']
        if not np.allclose(row['trace']['time'],epochs,atol=1e-11,rtol=0):raise ValueError('Complete diagnostic time grid required')
        # Native diagnostics are written before each step; snapshots0 and1 match trace indices0 and steps/2.
        trace_errors=[]
        for s,index in zip(row['snapshots'][:2],(0,row['spec']['steps']//2)):
            trace_errors.extend([abs(s['perturbation_kinetic_integral']-row['trace']['kinetic'][index]),
                abs(s['axial_nonzero_energy_integral']-(row['trace']['kinetic'][index]-row['trace']['axial_zero'][index])),
                abs(s['temperature_perturbation_energy_integral']-row['trace']['thermal_energy'][index])])
        native_ok=bool(native_ok and max(trace_errors)<limits['native_audit_absolute'])
        passed=bool(passed and native_ok and all(v['spatial_modes_present'] for v in checks))
        audits[name]=dict(independent_native_agreement=native_ok,maximum_native_error=max(differences),maximum_trace_error=max(trace_errors),checks=checks,
            maximum_divergence_l2=max(s['divergence_l2'] for s in row['snapshots']),final_snapshot=row['snapshots'][-1])
        print('spatial audit',name,'native agreement',native_ok,flush=True)
    atomic_json(out/'audits.json',seal(audits))
    comparisons=fixture['comparisons'];verdicts={}
    for name,limit in [('restart',limits['restart_rms']),('time_half',limits['time_rms']),('fourier_fine',limits['space_rms']),('radial_fine',limits['space_rms'])]:
        verdicts[name]=dict(passed=comparisons[name]['maximum_rms']<limit,limit=limit,result=comparisons[name])
    atomic_json(out/'comparisons.json',seal(dict(results=verdicts,restart_prefix=fixture['restart_prefix'])))
    statistics={}
    for name,row in fixture['records'].items():
        trace=row['trace'];volume=2*np.pi**2/row['snapshots'][0]['metadata']['alpha']
        series=dict(axial_spatial_energy=(np.array(trace['kinetic'])-trace['axial_zero'])/volume,
            darcy_friction=32*np.array(trace['friction_velocity'])**2,reported_axial_temperature_gradient=trace['axial_gradient'])
        statistics[name]=analyse_window(trace['time'],series,WindowPolicy(**p['statistics_policy']))
    atomic_json(out/'statistics.json',seal(statistics))
    physical=bool(all(v['passed'] for v in verdicts.values()) and all(c['means_passed'] and c['walls_passed'] and c['continuity_passed'] and c['tails_passed'] for a in audits.values() for c in a['checks']))
    statistical=bool(all(v['qualified'] for v in statistics.values()))
    if source!=identity():raise ValueError('Spatial study source changed')
    report=dict(passed=passed,source=source,source_unchanged=True,external_cases=len(audits),real_spatial_modes_present=True,
        maximum_native_audit_error=max(v['maximum_native_error'] for v in audits.values()),maximum_trace_error=max(v['maximum_trace_error'] for v in audits.values()),
        maximum_divergence_l2=max(v['maximum_divergence_l2'] for v in audits.values()),
        native_restart_prefix_exact=fixture['restart_prefix']['physical_fields_bitwise_equal'],
        native_restart_fields_exact=comparisons['restart']['physical_fields_bitwise_equal'],
        maximum_restart_rms=comparisons['restart']['maximum_rms'],maximum_time_rms=comparisons['time_half']['maximum_rms'],
        maximum_fourier_rms=comparisons['fourier_fine']['maximum_rms'],maximum_radial_rms=comparisons['radial_fine']['maximum_rms'],
        integration_gates=verdicts,numerical_resolution_qualified=physical,thermal_statistics_qualified=statistical,
        turbulent_heat_transfer_qualified=bool(physical and statistical),duration=.2,time_unit='R/(2Ub)',
        station_usable=False,validated=False,npp_11_03_accepted=False,r1_accepted=False,closed_tasks=[],
        scope='Real spatial execution and audit qualified; short spinup and unresolved numerical/statistical gates are not DNS acceptance')
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Spatial software/audit qualification failed; evidence preserved')
    if reference and {q.name:sha(q) for q in out.glob('*.json')}!={q.name:sha(q) for q in Path(reference).glob('*.json')}:
        raise ValueError('Installed spatial scientific JSON differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,a.reference);print(json.dumps(dict(passed=r['passed'],thermal_statistics_qualified=r['thermal_statistics_qualified'])))

if __name__=='__main__':main()
