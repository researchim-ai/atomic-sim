import { mkdir, cp, writeFile, rm } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { content, languages } from '../src/content.mjs';
import { render } from '../src/render.mjs';
import { topics, chapters, ui } from '../src/physics/content.mjs';
import { renderPhysicsIndex, renderPhysicsChapter } from '../src/physics/render.mjs';
export const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const output = resolve(root, 'dist');
function keys(value, prefix = '') {
  return Object.entries(value).flatMap(([key, item]) => {
    const p = `${prefix}.${key}`;
    return item !== null && typeof item === 'object' ? keys(item, p) : [p];
  }).sort();
}
export async function build() {
  if (JSON.stringify(keys(content.ru)) !== JSON.stringify(keys(content.en))) throw new Error('Translation keys differ');
  for (const topic of topics) if (JSON.stringify(keys(chapters[topic].ru)) !== JSON.stringify(keys(chapters[topic].en))) throw new Error('Physics translations differ: '+topic);
  if (JSON.stringify(keys(ui.ru)) !== JSON.stringify(keys(ui.en))) throw new Error('Physics UI translations differ');
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  await cp(resolve(root, 'public'), output, { recursive: true });
  for (const lang of languages) {
    await mkdir(resolve(output, lang), { recursive: true });
    await writeFile(resolve(output, lang, 'index.html'), render(lang));
    await mkdir(resolve(output,lang,'physics'),{recursive:true});
    await writeFile(resolve(output,lang,'physics','index.html'),renderPhysicsIndex(lang));
    for (const topic of topics) {
      await mkdir(resolve(output,lang,'physics',topic),{recursive:true});
      await writeFile(resolve(output,lang,'physics',topic,'index.html'),renderPhysicsChapter(lang,topic));
    }
  }
  await writeFile(resolve(output, 'index.html'), render('ru', {root: true}));
  await writeFile(resolve(output, '404.html'), `<!doctype html><html lang="en"><meta charset="UTF-8"><meta name="viewport" content="width=device-width"><title>404 — Atomic Sim</title><body style="background:#f4f4eb;color:#233c30;font:18px system-ui;padding:10vw"><p>ATOMIC SIM / 404</p><h1>Здесь пока пусто.<br>Nothing here yet.</h1><p><a href="/ru/">Русская версия</a> · <a href="/en/">English version</a></p></body></html>`);
  console.log('Built Atomic Sim: RU + EN, homepages + physics guide (6 chapters) → web/dist');
}
if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await build();
