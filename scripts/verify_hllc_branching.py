"""Branched liquid HLLC: loops, flow reversal, axis covariance and late replay.

Stationary finite-volume pressure nodes and scalar local-axis reactions;
not a3D junction, calibrated loss model or experimental validation.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.model import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_network as network
from scripts import verify_hllc_long as histories
from scripts import long_hllc_reference as independent
from scripts import verify_heterogeneous_network as peer

MATRIX=(('branched_rest',2),('branch_reversal',2),('branch_reversal',4))
CAPS=(5e-6,2.5e-6)
TIMES=(0.,.0001,.00025,.0005,.001,.002)
RESTART_INDEX=4
REFERENCE_CAP=2e-5
SUPPORT_ATOL=1e-9
ORIENTATION_BUDGETS={k:v for k,v in peer.REFERENCE_BUDGETS.items() if k!='support_kg_m_s'}
ORIENTATION_BUDGETS['face_momentum_kg_m_s']=1e-8


def _finite(values):
    try:return all(not isinstance(v,bool) and math.isfinite(v) for v in values)
    except (TypeError,ValueError,OverflowError):return False


def request_for(case,cells,cap):
    if (case,cells) not in MATRIX or cap not in CAPS:raise ValueError('Declared branching matrix required')
    values={'a':(4.04e6,441.,.003),'j':(4e6,440.,.0015),'b':(3.98e6,439.,.002),'c':(4.01e6,442.,.0025)}
    def initial(p,t,v=0.):return dict(kind='pt',pressure_pa=p,temperature_k=t,velocity_m_s=v)
    nodes=[dict(id=name,volume_m3=v,initial=initial(p,t),heat_power_w=0.) for name,(p,t,v) in values.items()]
    edges=[]
    for name,a,b,length,diameter in [('aj','a','j',1.2,.05),('jb','j','b',.9,.04),('cj','c','j',1.4,.03),('ba','b','a',2.,.025),('ca','c','a',1.7,.02)]:
        pa,ta,_=values[a];pb,tb,_=values[b]
        points=[initial(pa+(pb-pa)*(i+.5)/cells,ta+(tb-ta)*(i+.5)/cells,-.005 if case=='branch_reversal' and name=='jb' else 0.) for i in range(cells)]
        edges.append(dict(id=name,from_node=a,to_node=b,cells=cells,length_m=length,diameter_m=diameter,roughness_m=0.,friction='none',initial=points,heat_power_w=[0.]*cells))
    walls=[dict(id='wall',cell='jb/0',capacity_j_k=1e4,initial_temperature_k=440.,conductance_w_k=0.,heat_power_w=0.,ambient_ua_w_k=0.,ambient_temperature_k=440.)]
    return dict(schema_version=1,friction={},model=dict(schema_version=1,nodes=nodes,edges=edges,walls=walls,wave_speed_bound_m_s=2500.,max_step_s=cap,cfl=.3))


def reverse_edge(request,name):
    result=deepcopy(request);matches=[e for e in result['model']['edges'] if e['id']==name]
    if len(matches)!=1:raise ValueError('Unique edge required')
    edge=matches[0];edge['from_node'],edge['to_node']=edge['to_node'],edge['from_node']
    edge['initial']=list(reversed(edge['initial']));edge['heat_power_w'].reverse()
    for point in edge['initial']:point['velocity_m_s']=-point['velocity_m_s']
    for wall in result['model']['walls']:
        if wall['cell'].startswith(name+'/'):
            index=int(wall['cell'].split('/')[1]);wall['cell']=f'{name}/{edge["cells"]-1-index}'
    return result


def branch_velocity(snapshot,name):
    cells=[c for c in snapshot['cells'] if c['id'].startswith(name+'/')]
    if not cells:raise ValueError('Branch absent from observations')
    if any(not _finite((c['mass_kg'],c['momentum_kg_m_s'])) or c['mass_kg']<=0 for c in cells):raise ValueError('Positive finite branch stocks required')
    return math.fsum(c['momentum_kg_m_s'] for c in cells)/math.fsum(c['mass_kg'] for c in cells)


def audit(native):
    histories.history.validate_times([s['time_s'] for s in native['observations']])
    request=native['request'];nodes=len(request['model']['nodes']);first=native['observations'][0]
    names,_,faces,_=network.topology(request,first);maximum=0.;momentum=0.;velocities=[]
    for snapshot in native['observations']:
        if [c['id'] for c in snapshot['cells']]!=names:raise ValueError('Cell identity changed')
        ledger=snapshot['local_ledgers'];face_momentum=ledger['integrated_face_momentum_kg_m_s'];support=ledger['node_support_impulse_kg_m_s']
        if len(face_momentum)!=len(faces) or len(support)!=len(names):raise ValueError('Incomplete support/face ledger')
        if not _finite([*face_momentum,*support]):raise ValueError('Nonfinite support/face ledger')
        if any(not _finite((c['mass_kg'],c['momentum_kg_m_s'])) or c['mass_kg']<=0 for c in snapshot['cells']):raise ValueError('Positive finite stocks required')
        expected=[0.]*len(names)
        for value,(a,b) in zip(face_momentum,faces,strict=True):
            if a<nodes:expected[a]+=value
            if b<nodes:expected[b]-=value
        maximum=max(maximum,max(abs(a-b) for a,b in zip(expected,support,strict=True)))
        momentum=max(momentum,abs(math.fsum(c['momentum_kg_m_s']-c0['momentum_kg_m_s']-s for c,c0,s in zip(snapshot['cells'],first['cells'],support,strict=True))))
        if any(c['momentum_kg_m_s']!=0. for c in snapshot['cells'][:nodes]):raise ValueError('Stationary node acquired momentum')
        velocities.append(branch_velocity(snapshot,'jb'))
    checks=dict(port_support_balance=maximum<=SUPPORT_ATOL,signed_axial_balance=momentum<=SUPPORT_ATOL)
    return dict(checks=checks,passed=all(checks.values()),maximum_support_error_kg_m_s=maximum,maximum_axial_residual_kg_m_s=momentum,branch_velocity_m_s=velocities)


def orientation(before,after,edge_name):
    if after['request']!=reverse_edge(before['request'],edge_name):raise ValueError('Orientation request mismatch')
    aa=histories.history.validate_times([s['time_s'] for s in before['observations']])
    bb=histories.history.validate_times([s['time_s'] for s in after['observations']])
    if len(aa)!=len(bb) or any(abs(a-b)>1e-12 for a,b in zip(aa,bb,strict=True)):raise ValueError('Orientation time grid mismatch')
    names_a,_,faces_a,_=network.topology(before['request'],before['observations'][0])
    names_b,_,faces_b,_=network.topology(after['request'],after['observations'][0])
    if len(faces_a)!=len(faces_b):raise ValueError('Orientation face count mismatch')
    errors={k:0. for k in ORIENTATION_BUDGETS}
    edges=before['request']['model']['edges'];counts={e['id']:e['cells'] for e in edges}
    for a,b in zip(before['observations'],after['observations'],strict=True):
        if [c['id'] for c in a['cells']]!=names_a or [c['id'] for c in b['cells']]!=names_b:raise ValueError('Orientation cell identity mismatch')
        for observation in (a,b):
            for cell in observation['cells']:
                if not _finite(cell[field] for field in network.COLUMNS):raise ValueError('Nonfinite orientation cell')
            for key in ('integrated_face_mass_kg','integrated_face_energy_j','integrated_face_momentum_kg_m_s'):
                values=observation['local_ledgers'][key]
                if len(values)!=len(faces_a) or not _finite(values):raise ValueError('Incomplete/nonfinite orientation face ledger')
        mapped={c['id']:c for c in b['cells']}
        for cell in a['cells']:
            name=cell['id'];sign=1
            if name.startswith(edge_name+'/'):
                index=int(name.split('/')[1]);name=f'{edge_name}/{counts[edge_name]-1-index}';sign=-1
            other=mapped[name]
            for field in network.COLUMNS:
                expected=cell[field]*sign if field in ('momentum_kg_m_s','velocity_m_s') else cell[field]
                error=abs(expected-other[field])
                if not math.isfinite(error):raise ValueError('Nonfinite orientation cell error')
                errors[field]=max(errors[field],error)
        offset=0
        for edge in edges:
            size=edge['cells']+1
            for metric,key,sign in [('face_mass_kg','integrated_face_mass_kg',-1),('face_energy_j','integrated_face_energy_j',-1),('face_momentum_kg_m_s','integrated_face_momentum_kg_m_s',1)]:
                aa=a['local_ledgers'][key][offset:offset+size];bb=b['local_ledgers'][key][offset:offset+size]
                if edge['id']==edge_name:bb=[sign*v for v in reversed(bb)]
                for x,y in zip(aa,bb,strict=True):
                    error=abs(x-y)
                    if not math.isfinite(error):raise ValueError('Nonfinite orientation face error')
                    errors[metric]=max(errors[metric],error)
            offset+=size
    if any(not math.isfinite(v) for v in errors.values()):raise ValueError('Nonfinite orientation errors')
    return dict(maximum_errors=errors,passed=all(v<=ORIENTATION_BUDGETS[k] for k,v in errors.items()))


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,network.__file__,histories.__file__,histories.history.__file__,independent.__file__,peer.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,matrix=MATRIX,caps_s=CAPS,times_s=TIMES,restart_time_s=TIMES[RESTART_INDEX],
        native_budgets=network.BUDGETS,reference_cross_budgets=peer.REFERENCE_BUDGETS,orientation_budgets=ORIENTATION_BUDGETS,support_atol=SUPPORT_ATOL,
        reference_methods=('Radau','BDF'),reference_rtol=2e-11,reference_atol=1e-10,reference_max_step_s=REFERENCE_CAP,
        scope='Stationary1D pressure nodes,5heterogeneous edges,two graph cycles; scalar local-axis reactions,not3D junction forces',validated=False,plant_referenced=False)))
    records=[];selected=None
    for case,cells in MATRIX:
        folder=output/f'{case}-{cells}';folder.mkdir()
        try:
            request=request_for(case,cells,CAPS[-1]);refs=[]
            for method in ('Radau','BDF'):
                refs.append(independent.solve(request,TIMES,method,max_step=REFERENCE_CAP,on_window=lambda w:atomic_json(folder/f'{method}-progress.json',seal(w))))
            atomic_json(folder/'references.json',seal(dict(references=refs)))
            cross=histories.history.compare(histories.reference_packet(request,refs[0]),peer.as_peer(refs[1]['observations']),times=TIMES,budgets=peer.REFERENCE_BUDGETS)
            runs=[];comparisons=[]
            for cap in CAPS:
                native=histories.capture(request_for(case,cells,cap),times=TIMES,restart_index=RESTART_INDEX)
                ledger=audit(native);vel=ledger['branch_velocity_m_s']
                checks=dict(native=native['record']['passed'],support=ledger['passed'])
                if case=='branch_reversal':checks['flow_reversal']=vel[0]<-1e-3 and max(vel[1:])>1e-3
                for ref in refs:comparisons.append(dict(cap_s=cap,method=ref['method'],**histories.history.compare(native,peer.as_peer(ref['observations']),times=TIMES)))
                runs.append(dict(cap_s=cap,checks=checks,audit=ledger,passed=all(checks.values())))
                atomic_json(folder/f'native-{cap}.json',seal(native))
                if case=='branch_reversal' and cells==2 and cap==CAPS[-1]:selected=native
                print(case,cells,cap,runs[-1]['passed'],vel,flush=True)
            record=dict(case=case,cells=cells,runs=runs,comparisons=comparisons,reference_cross=cross,
                passed=cross['passed'] and all(r['passed'] for r in runs) and all(c['passed'] for c in comparisons if c['cap_s']==CAPS[-1]))
        except (ValueError,RuntimeError,OSError,KeyError) as error:record=dict(case=case,cells=cells,passed=False,error=str(error))
        records.append(record);atomic_json(folder/'report.json',seal(record));atomic_json(output/'progress.json',seal(dict(completed=False,records=records)))
        print('case',case,cells,record['passed'],record.get('error'),flush=True)
    oriented=dict(passed=False,error='Selected trajectory unavailable')
    if selected is not None:
        try:
            request=reverse_edge(selected['request'],'jb');native=histories.capture(request,times=TIMES,restart_index=RESTART_INDEX)
            ledger=audit(native);comparison=orientation(selected,native,'jb')
            model=ContactHEMNetwork(request,scheme='hllc-local');before=model.checkpoint()
            try:model.restore(selected['checkpoint'])
            except ValueError:refused=model.checkpoint()==before
            else:refused=False
            oriented=dict(comparison=comparison,audit=ledger,cross_request_checkpoint_refused=refused,
                passed=comparison['passed'] and ledger['passed'] and native['record']['passed'] and refused)
            atomic_json(output/'orientation-native.json',seal(native))
        except (ValueError,RuntimeError,OSError,KeyError) as error:oriented=dict(passed=False,error=str(error))
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,passed=unchanged and len(records)==len(MATRIX) and all(r['passed'] for r in records) and oriented['passed'],
        source_unchanged=unchanged,records=records,orientation=oriented,validated=False,plant_referenced=False,whole_plant_compared=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    return 0 if run(p.parse_args().output)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
