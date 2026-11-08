import { heatingRate, phaseFractions } from './physics-math.js';
const lang=document.documentElement.lang;
const format=(n,places=1)=>new Intl.NumberFormat(lang,{minimumFractionDigits:places,maximumFractionDigits:places}).format(n);
const heatIn=document.querySelector('#heat-in'),heatOut=document.querySelector('#heat-out');
if(heatIn&&heatOut){
 heatIn.disabled=false;heatOut.disabled=false;
 const states=lang==='ru'?['Тело нагревается','Баланс: температура не меняется','Тело остывает']:['The body heats up','Balanced: temperature is constant','The body cools down'];
 function update(){
  const rate=heatingRate(Number(heatIn.value),Number(heatOut.value));
  document.querySelector('#heat-in-value').value=`${format(Number(heatIn.value),0)} kW`;
  document.querySelector('#heat-out-value').value=`${format(Number(heatOut.value),0)} kW`;
  document.querySelector('#heat-rate').value=`${rate>0?'+':''}${format(rate,2)} K/s`;
  document.querySelector('#heat-state').textContent=states[rate>0?0:rate===0?1:2];
 }
 heatIn.addEventListener('input',update);heatOut.addEventListener('input',update);update();
}
const quality=document.querySelector('#quality');
if(quality){
 quality.disabled=false;
 function update(){
  const x=Number(quality.value)/100,r=phaseFractions(x);
  document.querySelector('#quality-value').value=`${format(x*100)}%`;
  document.querySelector('#void-fraction').value=`${format(r.voidFraction*100)}%`;
  document.querySelector('#mixture-density').value=`${format(r.density)} kg/m³`;
  document.querySelector('#vapour-volume').style.width=`${r.voidFraction*100}%`;
 }
 quality.addEventListener('input',update);update();
}
const links=[...document.querySelectorAll('.reading-nav nav a[href^="#"]')];
if('IntersectionObserver' in window){
 const observer=new IntersectionObserver(entries=>{
  const entry=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top)[0];
  if(!entry)return;
  for(const link of links){if(link.hash===`#${entry.target.id}`)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');}
 },{rootMargin:'-8% 0px -55% 0px'});
 for(const link of links){const section=document.getElementById(link.hash.slice(1));if(section)observer.observe(section);}
}
