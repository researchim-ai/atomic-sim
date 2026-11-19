import {parseTrace,MAX_BYTES,signedFlow,readingStatus,number} from './station-view-model.js';
const $=id=>document.getElementById(id),t=JSON.parse($('translations').textContent),lang=document.documentElement.lang;
let recording=null,original=null,index=0,timer=null,request=0;
const show=(id,value)=>{$(id).textContent=value;};
function element(tag,text,className){const e=document.createElement(tag);e.textContent=text;if(className)e.className=className;return e;}
function rows(id,values){const dl=document.createElement('dl');for(const [label,value]of values){dl.append(element('dt',label),element('dd',value));}$(id).replaceChildren(dl);}
const bool=x=>x?t.on:t.off;
function stop(){clearInterval(timer);timer=null;show('play',t.play);$('play').setAttribute('aria-pressed','false');}
function draw(){
 const f=recording.frames[index];$('frame').value=index;show('clock',number(f.time_s,lang,'s'));show('frame-count',`${index+1} / ${recording.frames.length}`);
 $('flow-values').replaceChildren();
 for(const [id,x]of Object.entries(f.flows)){
  const flow=signedFlow(x),card=element('div','','flow-card');card.append(element('span',t.flows[id]),element('strong',number(x.value,lang,x.unit)),element('small',`${t.nodes[flow.source]} → ${t.nodes[flow.target]}`));$('flow-values').append(card);
  const p=$('flow-'+id);p.classList.toggle('inactive',x.value===0);p.setAttribute('marker-start',x.value<0?'url(#arrow)':'none');p.setAttribute('marker-end',x.value>0?'url(#arrow)':'none');
 }
 const e=f.equipment;
 rows('equipment',[...Object.entries(e.effective_drives).map(([name,active])=>[`${t.drives[name]??name} · ${t.effective}`,bool(active)]),...Object.entries(e.requested).map(([name,active])=>[`${t.drives[name.replace(/_on$/,'')]??(name==='dhr_requested'?'DHR':name)} · ${t.requested}`,bool(active)]),[t.primarySpeed,number(e.primary_speed_rad_s,lang,'rad/s')],[t.turbineSpeed,number(e.turbine_speed_rad_s,lang,'rad/s')],[t.inlet,number(e.steam_inlet_position,lang)],[t.rod,number(e.rod_position,lang)],[t.feedCheck,bool(e.feed_check_open)],[t.condensateCheck,bool(e.condensate_check_open)],[t.grid,bool(e.grid_available)],[t.dhr,bool(e.dhr_active)]]);
 const p=f.protections;
 rows('protections',[[t.trip,bool(p.trip_latched)],[t.pressure,bool(p.pressure_interlock)],[t.scram,bool(p.scram_latched)],[t.turbineTrip,bool(p.turbine_trip_latched)]]);
 $('protections').classList.toggle('latched',p.trip_latched||p.scram_latched||p.turbine_trip_latched);
 const alarms=[...p.trip_reasons.map(code=>{const key=code.split(':')[0];return t.reasonLabels[key]?`${t.reasonLabels[key]} [${code}]`:code;}),...p.alarms.map(a=>typeof a==='string'?a:JSON.stringify(a))];$('alarms').replaceChildren(...(alarms.length?alarms:[t.empty]).map(a=>element('p',a,'alarm')));
 $('measurements').replaceChildren();
 for(const [name,r]of Object.entries(f.measurements)){
  const tr=document.createElement('tr'),status=readingStatus(r);tr.className=status;
  for(const value of [`${t.channels[name]??name} [${name}]`,number(r.value,lang,r.unit),`${status==='good'?t.good:t.untrusted} [${r.quality.join(', ')}]`,number(r.age_s,lang),`${number(r.sampled_at_s,lang)} / ${number(r.available_at_s,lang)}`])tr.append(element('td',value));
  $('measurements').append(tr);
 }
 const b=f.balances;
 rows('balances',[[t.energy,b.energy_relative.toExponential(5)],[t.mass,b.mass_relative.toExponential(5)],[t.boron,b.boron_relative.toExponential(5)],[t.residual,number(b.energy_residual_j,lang)],[t.defect,number(b.instantaneous_energy_defect_w,lang)]]);
 $('events').replaceChildren(...(f.events.length?f.events.map(e=>element('li',`${number(e.time_s,lang,'s')} · ${e.kind} · ${JSON.stringify(Object.fromEntries(Object.entries(e).filter(([k])=>!['kind','time_s'].includes(k))))}`)):[element('li',t.empty)]));
}
async function open(text,version=++request){
 try{
  const next=await parseTrace(text);if(version!==request)return;
  stop();recording=next;original=text;index=0;$('frame').max=next.frames.length-1;
  for(const id of ['frame','play','download'])$(id).disabled=false;
  show('source',next.source.station_sha256);show('status',t.ready);$('status').classList.remove('error');draw();
 }catch(error){if(version!==request)return;show('status',t.error+' '+error.message);$('status').classList.add('error');}
}
$('trace-file').addEventListener('change',async event=>{
 const file=event.target.files[0];if(!file)return;
 if(file.size>MAX_BYTES){++request;show('status',t.error+' 8 MiB');$('status').classList.add('error');return;}
 const version=++request;
 try{await open(await file.text(),version);}catch{if(version===request)show('status',t.error);}
});
$('frame').addEventListener('input',()=>{stop();index=Number($('frame').value);draw();});
$('play').addEventListener('click',()=>{
 if(timer){stop();return;}if(index===recording.frames.length-1)index=0;draw();show('play',t.pause);$('play').setAttribute('aria-pressed','true');
 timer=setInterval(()=>{if(index+1>=recording.frames.length){stop();return;}index++;draw();},650);
});
$('download').addEventListener('click',()=>{
 const url=URL.createObjectURL(new Blob([original],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='station-recording.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
window.addEventListener('pagehide',stop);
fetch('../../station-demo.json',{cache:'no-cache'}).then(r=>{if(!r.ok)throw Error('Demo unavailable');return r.text();}).then(text=>{if(request===0)return open(text);}).catch(()=>{if(request===0)show('status',t.error);});
