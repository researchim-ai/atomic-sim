import test from 'node:test';
import assert from 'node:assert/strict';
import { content, snippets } from '../src/content.mjs';
import { render, escape } from '../src/render.mjs';
import { build, output } from '../scripts/build.mjs';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

test('both routes have translated content, metadata and live destinations', () => {
  for (const language of ['ru','en']) {
    const html=render(language);
    assert.match(html,new RegExp(`<html lang="${language}">`));
    assert.ok(html.includes(escape(content[language].description)));
    assert.equal((html.match(/<h1\b/g)||[]).length,1);
    for (const id of ['models','agents','start']) assert.ok(html.includes(`id="${id}"`));
    assert.ok(html.includes('hreflang="ru"') && html.includes('hreflang="en"'));
    assert.ok(!html.includes('href="#"'));
    assert.ok(!html.includes('undefined'));
    assert.equal(content[language].scenarios.length,3);
    assert.ok(html.includes(escape(content[language].illustration)));
  }
});
test('code examples retain the actual public API and observation guard', () => {
  for (const language of ['ru','en']) {
    const code=snippets(content[language]);
    assert.match(code.python,/observation_id=observation\["observation_id"\]/);
    assert.match(code.python,/Episode\.restore\(saved\)/);
    assert.match(code.gym,/from atomic_pwr_agent.gymnasium import ControlledPWREnv/);
    assert.match(code.llm,/create_controlled_episode/);
  }
});
test('static output includes both locales, root fallback and self-hosted fonts', async () => {
  await build();
  for (const lang of ['ru','en']) {
    const html=await readFile(resolve(output,lang,'index.html'),'utf8');
    for(const asset of ['styles.css','app.js','fonts/lato-heavy.woff2']){
      assert.ok((await stat(resolve(output,asset))).size > 0);
    }
    assert.ok(html.includes('../styles.css'));
  }
  const root=await readFile(resolve(output,'index.html'),'utf8');
  assert.ok(root.includes('href="./styles.css"'));
  assert.ok(root.includes('href="./en/"'));
});

import { topics, chapters } from '../src/physics/content.mjs';
import { renderPhysicsIndex, renderPhysicsChapter } from '../src/physics/render.mjs';
import { heatingRate, phaseFractions } from '../public/physics-math.js';
import { readdir } from 'node:fs/promises';

test('physics chapters have bilingual teaching material and existing project references', async () => {
  for(const lang of ['ru','en']){
    const index=renderPhysicsIndex(lang);
    for(const topic of topics){
      const c=chapters[topic][lang], html=renderPhysicsChapter(lang,topic);
      assert.equal((html.match(/<h1\b/g)||[]).length,1);
      assert.ok(c.sections.length>=3 && c.vars.length>=4);
      assert.ok(c.boundary.length>100 && c.implementation.length>100);
      assert.ok(html.includes(`href="../../../${lang==='ru'?'en':'ru'}/physics/${topic}/"`));
      assert.ok(index.includes(`href="${topic}/"`));
      assert.ok(!html.includes('undefined')&&!html.includes('href="#"'));
      if(lang==='en')assert.ok(!/[А-Яа-яЁё]/.test(html),topic+' English contains Russian');
      for(const path of [c.code,c.doc])assert.ok((await stat(resolve('..',path))).isFile(),path);
    }
  }
});
test('teaching balances preserve units, phase endpoints and mass/volume distinction', () => {
  assert.equal(heatingRate(100,80),1);
  assert.equal(heatingRate(100,100),0);
  assert.equal(heatingRate(0,100),-5);
  assert.deepEqual(phaseFractions(0),{density:900,voidFraction:0});
  assert.deepEqual(phaseFractions(1),{density:5,voidFraction:1});
  const r=phaseFractions(.01);
  assert.ok(Math.abs(r.density-1/(.99/900+.01/5))<1e-10);
  assert.ok(Math.abs(r.voidFraction-18/27.9)<1e-12);
  assert.throws(()=>phaseFractions(-.1),RangeError);
  assert.throws(()=>phaseFractions(1.1),RangeError);
  assert.throws(()=>heatingRate(1,0,0),RangeError);
});
test('navigation, translation links, anchors and assets resolve across all 18 HTML pages', async () => {
  const files=[];
  async function scan(dir){for(const item of await readdir(dir,{withFileTypes:true})){const p=resolve(dir,item.name);if(item.isDirectory())await scan(p);else if(p.endsWith('.html'))files.push(p);}}
  await scan(output);assert.equal(files.length,18);
  for(const file of files){
    const html=await readFile(file,'utf8');
    const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
    assert.equal(ids.length,new Set(ids).size,'duplicate id in '+file);
    for(const [,link] of html.matchAll(/(?:href|src)="([^"]+)"/g)){
      if(/^(?:https?:|data:|mailto:)/.test(link))continue;
      const u=new URL(link,'https://local.test/'+file.slice(output.length+1));
      let path=resolve(output,'.'+decodeURIComponent(u.pathname));
      if(u.pathname.endsWith('/'))path=resolve(path,'index.html');
      assert.ok((await stat(path)).isFile(),file+' → '+link);
      if(u.hash){const target=await readFile(path,'utf8');assert.ok(target.includes(`id="${decodeURIComponent(u.hash.slice(1))}"`),file+' → '+link);}
    }
  }
});
