'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="project-token"]').content;
let view = null;
let pending = null;
let busy = false;
let generation = 0;
let pendingData = null;
let evaluationTimer = null;
function setBusy(value) { busy=value; for(const el of document.querySelectorAll("input,select,textarea,button"))el.disabled=value; if(!value){$("commit").disabled=!pending;$("data-attach").disabled=!pendingData;$("restore").disabled=!view || view.revision===0;} document.querySelector("main").setAttribute("aria-busy",String(value)); }
function status(message, error=false) { $('status').textContent=message; $('status').classList.toggle('error',error); }
function invalidate() { generation++; pending=null; pendingData=null; $("data-attach").disabled=true; $('commit').disabled=true; $('diff').textContent=''; $('proposal').textContent='Проверьте текущие значения перед сохранением.'; }
async function api(path, value) {
  const options={headers:{'X-Project-Token':token}};
  if(value!==undefined) {options.method='POST'; options.headers['Content-Type']='application/json'; options.body=JSON.stringify(value);}
  const response=await fetch(path,options);
  const data=await response.json();
  if(!response.ok) throw new Error(data.error || 'Запрос отклонён');
  return data;
}
function options(id, names, selected) { const el=$(id); el.replaceChildren(); for(const name of names) {const o=document.createElement('option');o.value=name;o.textContent=name;el.append(o);} if(selected!==undefined && names.includes(selected))el.value=selected; }
function showScenario() {
  if(!view)return;invalidate();const name=$('scenario').value;const s=view.project.scenarios[name];
  options('preset',view.model.presets,s.config.preset);$('dt').value=s.config.action_dt_s;$('horizon').value=s.config.episode_horizon_s;$('step').value=s.config.max_step_s;
  options('sensor',view.channels[name].observations.map(c=>c.name),s.objective.sensor);$('target').value=s.objective.target_k;$('tolerance').value=s.objective.tolerance_k;
  $('channels').textContent=JSON.stringify(view.channels[name],null,2);
}
function showPolicy(){if(!view)return;invalidate();$('policy-json').value=JSON.stringify(view.project.policies[$('policy').value],null,2);}
async function refresh(){
  const oldScenario=$('scenario').value;const oldPolicy=$('policy').value;view=await api('/api/inspect');invalidate();
  $('model').textContent=view.model.model_id+' · '+view.model.status+' · validated='+view.model.validated+' · plant_referenced='+view.model.plant_referenced;
  $('limits').textContent=view.model.applicability+'; '+view.model.approximations.join('; ');
  $('provenance').textContent=JSON.stringify(view.source,null,2);
  $('revision').textContent='Ревизия '+view.revision+' · '+view.project.project_id+' · '+view.project_sha256;
  options('scenario',Object.keys(view.project.scenarios),oldScenario);options('policy',Object.keys(view.project.policies),oldPolicy);showScenario();showPolicy();
  $('history').replaceChildren();for(const row of view.history){const li=document.createElement('li');li.textContent='№'+row.revision+' · '+row.operation+' · '+row.note;$('history').append(li);}
  options('restore-target',view.history.filter(r=>r.revision<view.revision).map(r=>String(r.revision)));$('restore').disabled=view.revision===0;
  $('study-panel').hidden=!view.study;if(view.study){options('data-binding',Object.keys(view.references));$('data-references').textContent=JSON.stringify(view.references,null,2);if($('evaluation-id').value){const state=await evaluationStatus(false);if(state.stale){$('evaluation-report').textContent='';$('evaluation-plot').replaceChildren();$('curve-caption').textContent='Сохранённый расчёт относится к другой ревизии (stale=true). Запросите закреплённый отчёт или новый расчёт.';}}}
  $('results-panel').hidden=!view.results_enabled;if(view.results_enabled){options('results-selection',view.result_evaluations.map(r=>r.evaluation_id));for(const option of $('results-selection').options){const row=view.result_evaluations.find(r=>r.evaluation_id===option.value);option.textContent=row.evaluation_id.slice(0,8)+' · '+row.binding_id+' · '+row.status+' · revision '+row.study_revision+(row.stale?' · stale':'');} $('results-output').replaceChildren();$('results-message').textContent=view.result_list_truncated?'Показаны первые 64 ID в лексикографическом порядке; остальные доступны через CLI.':'';}
  status('Проект проверен. Выберите параметры для редактирования.');
}
function finite(id){const text=$(id).value;if(!text.trim())throw new Error('Числовое поле не заполнено');const value=Number(text);if(!Number.isFinite(value))throw new Error('Нужно конечное числовое значение');return value;}
function request(operation){return {schema_version:1,expected_revision:view.revision,expected_project_sha256:view.project_sha256,operations:[operation],note:$('note').value};}
async function preview(operation){const req=request(operation);const baseGeneration=generation;const result=await api('/api/preview',req);if(generation!==baseGeneration)throw new Error("Параметры изменились во время проверки. Повторите preview.");pending=req;$('diff').textContent=JSON.stringify(result.changes,null,2);$('proposal').textContent='Проверено. Новый digest: '+result.proposed_project_sha256;$('commit').disabled=false;status('Все сценарии и политики прошли проверку. Изменения ещё не сохранены.');}
function action(id, fn){$(id).addEventListener('click',async()=>{if(busy)return;setBusy(true);try{await fn();}catch(error){invalidate();status(error.message,true);}finally{setBusy(false);}});}
$('scenario').addEventListener('change',showScenario);$('policy').addEventListener('change',showPolicy);
for(const id of ['preset','dt','horizon','step','sensor','target','tolerance','policy-json','note'])$(id).addEventListener('input',invalidate);
action('refresh',refresh);
action('preview-scenario',async()=>{const scenario=structuredClone(view.project.scenarios[$('scenario').value]);scenario.config={preset:$('preset').value,action_dt_s:finite('dt'),episode_horizon_s:finite('horizon'),max_step_s:finite('step')};scenario.objective={sensor:$('sensor').value,target_k:finite('target'),tolerance_k:finite('tolerance')};await preview({op:'replace_scenario',id:$('scenario').value,value:scenario});});
action('preview-policy',()=>preview({op:'replace_policy',id:$('policy').value,value:JSON.parse($('policy-json').value)}));
action('commit',async()=>{if(!pending)throw new Error('Сначала проверьте изменения');await api('/api/edit',pending);await refresh();status('Новая ревизия сохранена.');});
action('restore',async()=>{await api('/api/restore',{expected_revision:view.revision,expected_project_sha256:view.project_sha256,target_revision:Number($('restore-target').value)});await refresh();status('Конфигурация восстановлена новой ревизией.');});
action('export',async()=>{const result=await api('/api/export',{});$('export-result').textContent='Проект: '+result.project;status('Экспорт проверен и сохранён.');});
setBusy(true);
refresh().catch(error=>status(error.message,true)).finally(()=>setBusy(false));

function studyCAS(){return {expected_revision:view.revision,expected_checkpoint_sha256:view.checkpoint_sha256};}
function evaluationRequest(){return {evaluation_id:$('evaluation-id').value};}
async function uploadData(){
  const metadata=$('data-declaration').files[0],csv=$('data-csv').files[0];
  if(!metadata || !csv)throw new Error('Выберите декларацию и CSV');
  if(metadata.size>1024*1024 || csv.size>8*1024*1024)throw new Error('Лимит: JSON 1 MiB, CSV 8 MiB');
  const declaration=JSON.parse(await metadata.text());const bytes=new Uint8Array(await csv.arrayBuffer());
  let binary='';for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.subarray(i,i+32768));
  return {request:{schema_version:1,...studyCAS(),id:$('data-id').value,scenario_id:$('scenario').value,policy_id:$('policy').value,declaration,note:$('note').value},csv_base64:btoa(binary)};
}
for(const id of ['data-id','data-declaration','data-csv'])$(id).addEventListener('input',invalidate);
action('data-preview',async()=>{const gen=generation;const payload=await uploadData();const result=await api('/api/data-preview',payload);if(gen!==generation)throw new Error('Данные изменились. Повторите проверку.');pendingData=payload;$('data-preview-result').textContent=JSON.stringify(result,null,2);status('Данные проверены; проект ещё не изменён.');});
action('data-attach',async()=>{if(!pendingData)throw new Error('Сначала проверьте данные');await api('/api/data-attach',pendingData);await refresh();status('Модель и эталон закреплены новой ревизией.');});
action('data-detach',async()=>{await api('/api/data-detach',{...studyCAS(),id:$('data-binding').value});await refresh();});
async function evaluationStatus(poll=true){
  clearTimeout(evaluationTimer);const id=$('evaluation-id').value;const result=await api('/api/evaluation-status',{evaluation_id:id});
  if(id!==$('evaluation-id').value)return;
  $('evaluation-state').textContent=JSON.stringify(result,null,2);
  if(poll && result.status==='running')evaluationTimer=setTimeout(()=>{if(busy){evaluationTimer=setTimeout(()=>evaluationStatus().catch(e=>status(e.message,true)),500);return;}evaluationStatus().catch(e=>status(e.message,true));},500);
  return result;
}
action('evaluation-start',async()=>{const result=await api('/api/evaluation-start',{...studyCAS(),binding_id:$('data-binding').value});$('evaluation-id').value=result.evaluation_id;$('evaluation-report').textContent='';$('evaluation-plot').replaceChildren();await evaluationStatus();});
action('evaluation-status',()=>evaluationStatus());
for(const operation of ['cancel','resume'])action('evaluation-'+operation,async()=>{await api('/api/evaluation-'+operation,evaluationRequest());await evaluationStatus();});
action('evaluation-summary',async()=>{const result=await api('/api/evaluation-summary',evaluationRequest());$('evaluation-report').textContent=JSON.stringify(result,null,2);options('evaluation-channel',Object.keys(result.channels));await evaluationStatus();});
function drawCurve(data){
  const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 800 300');svg.setAttribute('role','img');svg.setAttribute('aria-label','Модель — синий; эталон — оранжевый');
  const all=[...data.model.value,...data.reference.value];let low=Infinity,high=-Infinity;for(const v of all){low=Math.min(low,v);high=Math.max(high,v);}const span=high-low || Math.max(1,Math.abs(high)*0.01);const t0=data.model.time_s[0],t1=data.model.time_s.at(-1);
  for(const [key,color] of [['model','#2563eb'],['reference','#c2410c']]){const curve=data[key];const line=document.createElementNS(ns,'polyline');line.setAttribute('fill','none');line.setAttribute('stroke',color);line.setAttribute('stroke-width','2');line.setAttribute('points',curve.time_s.map((t,i)=>(40+720*(t-t0)/(t1-t0))+','+(260-220*(curve.value[i]-low)/span)).join(' '));svg.append(line);}
  for(const [x,y,label] of [[40,285,t0+' s'],[680,285,t1+' s'],[5,30,String(high)],[5,260,String(low)]]){const text=document.createElementNS(ns,'text');text.setAttribute('x',x);text.setAttribute('y',y);text.textContent=label;svg.append(text);}
  $('evaluation-plot').replaceChildren(svg);$('curve-caption').textContent=data.channel+' ['+data.unit+']; синий — модель, оранжевый — эталон. Линейная интерполяция; сохранены все узлы. Устарел относительно HEAD: '+data.stale;
}
action('evaluation-curve',async()=>drawCurve(await api('/api/evaluation-curve',{...evaluationRequest(),channel:$('evaluation-channel').value})));

function selectedResults(){return {evaluation_ids:Array.from($('results-selection').selectedOptions,o=>o.value)};}
function compactMetric(value){const magnitude=Math.abs(value);return magnitude>0 && (magnitude<0.001 || magnitude>=10000)?value.toExponential(1):Number(value.toPrecision(3)).toString();}
function showComparison(result){
  const container=$('results-output');container.replaceChildren();
  const intro=document.createElement('p');intro.textContent='Расчётов: '+Object.keys(result.entries).length+'; групп сопоставимости: '+Object.keys(result.groups).length+'. Время и RSS не используются для рейтинга.';container.append(intro);
  for(const [groupId,group] of Object.entries(result.groups)){
    const section=document.createElement('section');const heading=document.createElement('h3');heading.textContent='Протокол '+groupId.slice(0,12);section.append(heading);
    for(const [channel,ranking] of Object.entries(group.channel_rankings)){
      const title=document.createElement('h4');title.textContent=channel;section.append(title);
      const table=document.createElement('table');const header=document.createElement('tr');for(const label of ['Ранг','Расчёт','Политика','MAE ≈ ('+ranking[0].unit+')']){const cell=document.createElement('th');cell.textContent=label;header.append(cell);}const thead=document.createElement('thead');thead.append(header);table.append(thead);const body=document.createElement('tbody');
      for(const row of ranking){const tr=document.createElement('tr');for(const [text,full] of [[row.rank,row.rank],[row.evaluation_id.slice(0,6),row.evaluation_id],[result.entries[row.evaluation_id].policy_id,result.entries[row.evaluation_id].policy_id],[compactMetric(row.mae),row.mae]]){const cell=document.createElement('td');cell.textContent=String(text);cell.title=String(full);tr.append(cell);}body.append(tr);}table.append(body);section.append(table);
    }container.append(section);
  }
  for(const [id,row] of Object.entries(result.entries)){if(!row.eligible){const text=document.createElement('p');text.textContent=id.slice(0,8)+' · '+row.status+' · '+row.exclusion+' · '+(row.reason||'');container.append(text);}}
}
action('results-refresh',refresh);
action('results-compare',async()=>{const result=await api('/api/results-compare',selectedResults());showComparison(result);$('results-message').textContent='Проверены native-траектории и пересчитаны метрики выбранных результатов.';});
action('results-export',async()=>{const result=await api('/api/results-export',selectedResults());$('results-message').textContent='Пакет результатов: '+result.bundle;});
