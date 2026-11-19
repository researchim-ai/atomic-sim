export const MAX_BYTES=8*1024*1024;
export const flowSpec={steam:['sg','turbine','kg/s'],feed:['feed','sg','kg/s'],condensate:['condenser','feed','kg/s'],condenser_heat:['condenser','cooling','W'],external_heat:['cooling','external','W'],gross:['turbine','bus','W'],export:['bus','grid','W'],auxiliary:['bus','auxiliary','W'],dhr_heat:['sg','cooling','W']};
const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const keys=(x,expected)=>object(x)&&Object.keys(x).sort().join('|')===[...expected].sort().join('|');
export function validate(body){
 if(!keys(body,['format','privileged','source','frames'])||body.format!=='station-view/v1'||body.privileged!==true)throw Error('Invalid privileged station-view/v1 document');
 if(!keys(body.source,['station_sha256','viewer_sha256'])||!Object.values(body.source).every(x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x)))throw Error('Invalid source hashes');
 if(!Array.isArray(body.frames)||body.frames.length<1||body.frames.length>1025)throw Error('Expected 1–1025 frames');
 let previous=-1;
 for(const f of body.frames){
  if(!keys(f,['time_s','flows','equipment','protections','measurements','balances','events'])||!finite(f.time_s)||f.time_s<0||f.time_s<=previous)throw Error('Invalid frame clock/schema');
  previous=f.time_s;
  if(!keys(f.flows,Object.keys(flowSpec)))throw Error('Invalid flow graph');
  for(const [id,[source,target,unit]] of Object.entries(flowSpec)){
   const x=f.flows[id];if(!keys(x,['source','target','unit','value'])||x.source!==source||x.target!==target||x.unit!==unit||!finite(x.value))throw Error('Invalid flow: '+id);
  }
  if(!keys(f.equipment,['requested','effective_drives','primary_speed_rad_s','turbine_speed_rad_s','feed_check_open','condensate_check_open','steam_inlet_position','rod_position','dhr_active','grid_available']))throw Error('Invalid equipment');
  for(const [group,names] of Object.entries({requested:['primary_pump_on','feed_pump_on','condensate_pump_on','cooling_pump_on','dhr_requested'],effective_drives:['primary_pump','feed_pump','condensate_pump','cooling_pump','pressurizer_heater']}))if(!keys(f.equipment[group],names)||!Object.values(f.equipment[group]).every(x=>typeof x==='boolean'))throw Error('Invalid drive flags');
  for(const name of ['primary_speed_rad_s','turbine_speed_rad_s','steam_inlet_position','rod_position'])if(!finite(f.equipment[name]))throw Error('Invalid equipment value');
  for(const name of ['feed_check_open','condensate_check_open','dhr_active','grid_available'])if(typeof f.equipment[name]!=='boolean')throw Error('Invalid equipment flag');
  const p=f.protections;
  if(!keys(p,['trip_latched','trip_reasons','pressure_interlock','scram_latched','turbine_trip_latched','alarms']))throw Error('Invalid protections');
  for(const name of ['trip_latched','pressure_interlock','scram_latched','turbine_trip_latched'])if(typeof p[name]!=='boolean')throw Error('Invalid protection flag');
  if(!Array.isArray(p.trip_reasons)||!p.trip_reasons.every(x=>typeof x==='string')||!Array.isArray(p.alarms))throw Error('Invalid protection reasons');
  if(!object(f.measurements)||Object.keys(f.measurements).length>64)throw Error('Invalid measurements');
  for(const r of Object.values(f.measurements)){
   if(!keys(r,['value','unit','sampled_at_s','available_at_s','age_s','quality'])||!(r.value===null||finite(r.value))||typeof r.unit!=='string'||!Array.isArray(r.quality)||!r.quality.length||!r.quality.every(x=>typeof x==='string'))throw Error('Invalid reading');
   if(![r.sampled_at_s,r.available_at_s,r.age_s].every(finite)||r.age_s<0||r.sampled_at_s>r.available_at_s||r.available_at_s>f.time_s||Math.abs(r.age_s-(f.time_s-r.sampled_at_s))>1e-8)throw Error('Invalid measurement clock');
  }
  if(!keys(f.balances,['energy_relative','mass_relative','boron_relative','energy_residual_j','instantaneous_energy_defect_w'])||!Object.values(f.balances).every(finite)||['energy_relative','mass_relative','boron_relative'].some(k=>f.balances[k]<0))throw Error('Invalid balances');
  if(!Array.isArray(f.events)||f.events.length>10000||!f.events.every(e=>object(e)&&typeof e.kind==='string'&&finite(e.time_s)&&e.time_s>=0&&e.time_s<=f.time_s+1e-8))throw Error('Invalid events');
 }
 return body;
}
export async function parseTrace(text,cryptoApi=globalThis.crypto){
 if(typeof text!=='string'||new TextEncoder().encode(text).length>MAX_BYTES)throw Error('File exceeds 8 MiB');
 const e=JSON.parse(text);
 if(!keys(e,['format','document','sha256'])||e.format!=='station-view-file/v1'||typeof e.document!=='string'||typeof e.sha256!=='string'||!cryptoApi?.subtle)throw Error('Invalid envelope or unavailable SHA256 support');
 const hash=Array.from(new Uint8Array(await cryptoApi.subtle.digest('SHA-256',new TextEncoder().encode(e.document))),x=>x.toString(16).padStart(2,'0')).join('');
 if(hash!==e.sha256)throw Error('SHA256 mismatch');
 return validate(JSON.parse(e.document));
}
export function signedFlow(flow){return flow.value<0?{...flow,source:flow.target,target:flow.source,value:-flow.value}:flow;}
export function readingStatus(r){return r.value!==null&&r.quality.length===1&&r.quality[0]==='good'?'good':'untrusted';}
export function number(value,lang,unit=''){
 if(value===null)return '—';
 let scale=1,suffix=unit;
 if(unit==='W'){scale=1e6;suffix='MW';}
 return new Intl.NumberFormat(lang,{maximumSignificantDigits:6}).format(value/scale)+(suffix?' '+suffix:'');
}
