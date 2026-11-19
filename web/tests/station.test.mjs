import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto,createHash } from 'node:crypto';
import { parseTrace,validate,signedFlow,readingStatus,number,MAX_BYTES } from '../public/station-view-model.js';
import { renderStation,text } from '../src/station/render.mjs';
const file=new URL('../public/station-demo.json',import.meta.url);
const envelope=body=>{const document=JSON.stringify(body);return JSON.stringify({format:'station-view-file/v1',document,sha256:createHash('sha256').update(document).digest('hex')});};

test('actual Python export imports with verified UTF-8 digest and all five frames',async()=>{
 const b=await parseTrace(await readFile(file,'utf8'),webcrypto);
 assert.equal(b.frames.length,5);assert.equal(b.privileged,true);
 assert.equal(b.frames.at(-1).protections.scram_latched,true);
 assert.ok(b.frames.some(f=>readingStatus(f.measurements.pressure)==='untrusted'));
 assert.equal(readingStatus(b.frames[0].measurements.pressure),'good');
});
test('tampering, excessive files, bad hashes and unknown versions are refused',async()=>{
 const raw=await readFile(file,'utf8'),e=JSON.parse(raw);e.document+=' ';
 await assert.rejects(parseTrace(JSON.stringify(e),webcrypto),/SHA256/);
 await assert.rejects(parseTrace('x'.repeat(MAX_BYTES+1),webcrypto),/8 MiB/);
 await assert.rejects(parseTrace('{}',webcrypto),/envelope/);
 const b=await parseTrace(raw,webcrypto);b.format='other';await assert.rejects(parseTrace(envelope(b),webcrypto));
});
test('malformed clock, units, quality, equipment and nonfinite budgets fail validation',async()=>{
 const b=await parseTrace(await readFile(file,'utf8'),webcrypto);
 for(const mutate of [x=>x.frames[0].flows.steam.unit='MW',x=>x.frames[0].measurements.pressure.age_s=42,x=>x.frames[0].equipment.effective_drives.feed_pump='yes',x=>x.frames[0].balances.energy_relative=Infinity,x=>x.frames[0].events.push({time_s:4,kind:'future'}),x=>x.frames[0].measurements.pressure.quality=[],x=>x.frames.push(x.frames[0])]){const copy=structuredClone(b);mutate(copy);assert.throws(()=>validate(copy));}
});
test('signed import reverses orientation and does not fabricate trust',()=>{
 assert.deepEqual(signedFlow({source:'bus',target:'grid',unit:'W',value:-15}),{source:'grid',target:'bus',unit:'W',value:15});
 assert.equal(readingStatus({value:null,quality:['good']}),'untrusted');
 assert.equal(readingStatus({value:1,quality:['stale']}),'untrusted');
 assert.equal(number(null,'en'),'—');assert.equal(number(1e6,'en','W'),'1 MW');
});
test('bilingual viewer exposes local import, accessible timeline, measurements and limitations',()=>{
 for(const lang of ['ru','en']){
 const html=renderStation(lang);assert.equal((html.match(/<h1\b/g)||[]).length,1);
 for(const id of ['frame','clock','trace-file','measurements','equipment','protections','events','balances','flow-values'])assert.ok(html.includes(`id="${id}"`));
 assert.ok(html.includes('type="range"'));assert.ok(html.includes('role="status"'));assert.ok(html.includes('station-view.js'));
 assert.ok(html.includes(`../../${lang==='ru'?'en':'ru'}/station/`));assert.ok(!html.includes('undefined'));
 if(lang==='en')assert.ok(!/[А-Яа-яЁё]/.test(html));
 }
 assert.deepEqual(Object.keys(text.ru).sort(),Object.keys(text.en).sort());
 assert.throws(()=>renderStation('xx'));
});
test('file content never enters markup and automatic animation is absent',async()=>{
 const js=await readFile(new URL('../public/station-view.js',import.meta.url),'utf8');
 assert.ok(!/innerHTML|insertAdjacentHTML|eval\(/.test(js));assert.ok(js.includes('textContent'));
 assert.ok(!/localStorage|sessionStorage|XMLHttpRequest/.test(js));
 const css=await readFile(new URL('../public/station.css',import.meta.url),'utf8');assert.ok(css.includes('prefers-reduced-motion'));assert.ok(css.includes('focus-visible'));
});
