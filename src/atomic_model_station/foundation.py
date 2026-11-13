"""Versioned package-A passport, component reuse audit and acceptance inventory."""
from dataclasses import asdict
import hashlib
from importlib import import_module
from pathlib import Path
from .model import StationConfig

REUSE=[
 ('physics.hydraulics.network','adapter','Mass kg, energy J, boron kg, flow kg/s','Finite inventories; declared flow law and EOS per port','tests/test_thermal_hydraulics.py'),
 ('physics.neutronics.point_kinetics','adapter','Relative power and delayed precursor groups','Point model; coefficient family explicitly admitted','tests/test_physics_verification.py'),
 ('physics.thermal.fuel_rod','adapter','Radial fuel/clad temperatures K and powers W','Fuel geometry and closures per supplied configuration','tests/test_multiphysics.py'),
 ('atomic_model_hem_network','adapter','Mass/energy and node/edge unknowns in SI','Qualified HEM subdomains only; no general two-fluid claim','tests/test_hem_network.py'),
 ('core.residual','adapter','Index-1 blocks, dimensional rates','mixed unknowns/rank','tests/test_reduced_residual_engine.py'),
 ('atomic_model_controlled_plant.model','adapter','Finite fluid/thermal inventories; SI','Hot reduced-PWR, ideal feed sensing','tests/test_controlled_reduced_engine.py'),
 ('atomic_model_plant_extension.pwr','adapter','Point kinetics, fuels, fluid stocks; SI','Mixed provenance, not BEAVRS reconstruction','tests/test_plant_extension.py'),
 ('physics.power.steam_turbine_train','adapter','Rotor J, speed rad/s, power W','Operating band; fixed fluid boundaries','tests/test_steam_turbine_train.py'),
 ('components.condenser','adapter','Mass kg, energy J, EOS pressure Pa','Finite volume, admitted IF97 states','tests/test_exchanger_condenser.py'),
 ('components.heat_exchanger','ready','Wall K, capacity J/K, flows W','Local two-sided thermal balance; R0 adapter exercised','tests/test_exchanger_condenser.py'),
 ('atomic_model_plant_extension.water','adapter','p Pa, T K, rho kg/m3, u J/kg','Explicit IF97 domain; no whole-station admission','tests/test_plant_extension.py'),
 ('atomic_model_hem_closures','adapter','Friction/heat-transfer coefficient families','Each correlation domain and uncertainty separately','tests/test_hem_closure_audit.py'),
 ('atomic_model_neutronic_response','adapter','k, multigroup flux and derivatives 1/K','Steady absorption derivative only; no evaluated BEAVRS data','tests/test_neutronic_response.py'),
 ('atomic_pwr_agent','adapter','Observations, decisions, saved episodes','Reduced-PWR tools; station actions need explicit extension','tests/test_pwr_agent.py'),
 ('atomic_model_thermal_observer','redesign','Diagnostic history, assumed bounds, sample times','Known initial-bias/drift blind spots; no safety admission','tests/test_thermal_observer.py'),
]


def passport():
    return dict(schema_version=1,version='1.0.0',target_model_id='npp-pwr-research',
        executed_foundation_model_id='npp-r0-mixed-index1',
        equipment=['reactor','primary loops and pressurizer','steam generator','steam train','turbine rotor',
                   'generator active load','condenser','feedwater','cooling','auxiliary power'],
        foundation_scope='R0 index-1 thermal-mechanical test assembly; later equipment is not connected yet',
        boundaries=['prescribed heat input at R0 / reactor physics at R1','electrical grid export','external cooling','explicit makeup/drain at R1'],
        power_definitions=dict(shaft='Mechanical turbine output before generator losses',gross='Generator electrical output before own needs',
            own_needs='Explicit auxiliary electrical consumption',net='gross minus own_needs',export='Net power transferred across grid boundary'),
        initial_modes=['hot stationary operation','limited transitions','supported controlled shutdown'],
        unsupported=['Cold start without separate components','Severe accident physics','Full electromagnetic grid'],
        validated=False,plant_referenced=False)


def parameter_registry(config=None):
    c=config if config is not None else StationConfig()
    return dict(schema_version=1,model_version='0.1.0',parameters={k:dict(**v,version='1',uncertainty=None)
                for k,v in c.registry().items()},future_families=['illustrative','measured','calibrated'])


def reuse_audit():
    rows=[]
    for module,status,states,domain,test in REUSE:
        m=import_module(module);p=Path(m.__file__)
        rows.append(dict(module=module,status=status,states_and_units=states,applicability=domain,
            source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),test=test,
            data='Declared module passport/source and local test; no promotion to experimental validation'))
    return rows


def acceptance():
    return dict(version='1',levels={
        'R0':dict(required=['contracts','mixed map/rank','consistent initialization','load continuation','event root and topology reinitialization',
                           'global energy','queued commands','PI and sensor state','RNG','fresh-process JSON restart','rollback'],
                  targets=dict(initial_residual=1e-8,solver_scaled_residual=1e-10,energy_relative=1e-6,
                               reference_scaled_error=1e-3,event_localization_s=1e-6)),
        'R1':dict(required=['Connected physical reactor and steam/water cycle','NPP integrated matrix','partial-observation agent'],
                  targets=dict(energy_relative=1e-6,mass_relative=1e-10,refinement_relative=1e-3),accepted=False),
        'R2':dict(required=['Consistent plant conditions','Applicable measured uncertainties','Independent observable comparisons'],accepted=False),
        'R3':dict(required=['Qualified advanced closures and geometry for declared extended regimes'],accepted=False)})


def validate_registry(registry):
    if type(registry) is not dict or set(registry)!={'schema_version','model_version','parameters','future_families'} or registry['schema_version']!=1:raise ValueError('Registry schema/version')
    if type(registry['parameters']) is not dict or not 1<=len(registry['parameters'])<=256:raise ValueError('Parameter budget')
    import math
    for key,v in registry['parameters'].items():
        if type(key) is not str or not key or type(v) is not dict or set(v)!={'value','unit','source','family','version','uncertainty'}:raise ValueError('Parameter schema')
        if type(v['value']) not in (int,float) or not math.isfinite(v['value']) or v['family'] not in ('illustrative','measured','calibrated'):raise ValueError('Parameter value/family')
        if any(type(v[k]) is not str or not v[k] for k in ('unit','source','version')):raise ValueError('Parameter provenance/units')
        if v['uncertainty'] is not None and (type(v['uncertainty']) not in (int,float) or not math.isfinite(v['uncertainty']) or v['uncertainty']<0):raise ValueError('Uncertainty semantics')
    return True
