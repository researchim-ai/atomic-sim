"""Finite thermal case matrix with separately controlled numerical refinements."""
import argparse
from pathlib import Path

from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_thermal_cases import thermal_catalog,build_thermal_case,evaluate_thermal_matrix
from atomic_model_station_thermal_cases.provenance import case_identity
from atomic_model_station_thermal_cases.evaluate import CRITERIA,VARIANTS
from atomic_model_station_thermal_resolved import resolve_thermal,accept_resolved,ThermalQualityError


def identity():
    from scripts.qualify_station_thermal_path import identity as base_identity
    return dict(case_engine=case_identity(),driver=sha(__file__),base=base_identity())


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);cases=thermal_catalog()
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=dict(CRITERIA),horizon=.1,
        cases=[case.specification() for case in cases],variants=[list(row) for row in VARIANTS],
        units='Nondimensional constant-property single-phase Boussinesq anomaly',
        quadrature=128,comparison_quadrature=160,
        initial_field='Same seeded low-order velocity and analytic scalar field across all variants of each case',
        method='One uninterrupted real-split BDF per variant,identical start/end epochs;only declared numerical axes change',
        main_quality_variants=['main','time','fourier','temperature48'],
        coarse_radial_policy='Coarse comparison retained even if target heat/wall gate fails;coarse acceptance is not inferred',
        exclusions=['Parameter-box interpolation','Unseen-seed generalization','Statistical convergence','Long duration for all cases','Plant applicability','Experimental validation'])))
    def observer(row):
        print(row['case']['name'],'accepted',row['accepted'],'checks',row['checks'],flush=True)
    matrix=evaluate_thermal_matrix(cases,horizon=.1,on_case=observer)
    atomic_json(out/'matrix.json',seal(matrix))
    coarse_system=build_thermal_case(cases[0],band=1,velocity_order=10,temperature_order=10)
    coarse=resolve_thermal(coarse_system,end_time=.1)
    try:negative=accept_resolved(coarse_system,coarse)
    except ThermalQualityError as error:negative=error.receipt
    atomic_json(out/'counterexample.json',seal(dict(case=cases[0].specification(),space=dict(band=1,velocity_order=10,temperature_order=10),solver=coarse.report,quality=negative)))
    rows=matrix['cases']
    completed=sum(stage['completed'] for row in rows for stage in row['stages'].values())
    flat=next(row for row in rows if row['case']['name']=='flat_passive')
    checks=dict(finite_matrix=matrix['accepted'] and matrix['accepted_count']==5 and matrix['total']==5,
        all_variants_completed=completed==25,
        coarse_refusal=not negative['passed'],
        flat_wall_compatible=flat['stages']['main']['initial_wall']['compatible'],
        immutable_domain=not matrix['input_box_qualified'] and not matrix['interpolation_qualified'])
    main_stages=[row['stages']['main'] for row in rows]
    def norm(row):return max(row['velocity_l2'],row['temperature_l2'])
    if source!=identity():raise ValueError('Thermal case matrix source changed')
    report=dict(passed=bool(all(checks.values())),source=source,checks=checks,
        selected_case_matrix_qualified=bool(all(checks.values())),qualified_cases=matrix['accepted_count'],total_cases=5,
        completed_numerical_paths=completed,horizon=.1,
        maximum_main_heat_residual=max(stage['solver']['maximum_physical_heat_budget_residual'] for stage in main_stages),
        maximum_main_wall_residual=max(max(stage['quality']['audit']['maximum_wall_momentum_residual'],stage['quality']['audit']['maximum_wall_thermal_residual']) for stage in main_stages),
        maximum_temporal_l2=max(norm(row['comparisons']['time']) for row in rows),
        maximum_fourier_l2=max(norm(row['comparisons']['fourier']) for row in rows),
        maximum_radial_l2=max(norm(row['comparisons']['radial']) for row in rows),
        maximum_temperature_refinement_l2=max(norm(row['comparisons']['temperature48']) for row in rows),
        tested_case_specifications=matrix['tested_case_specifications'],
        physical_input_box_qualified=False,unseen_case_generalization=False,statistical_convergence_qualified=False,
        physical_duration_qualified=False,general_strong_pde_qualified=False,station_usable=False,validated=False,
        native_backend_changed=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Finite thermal matrix qualification failed;evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed thermal matrix scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();print(run(args.output,args.reference)['passed'])

if __name__=='__main__':main()
