import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { resolve, extname, relative, sep } from 'node:path';
import { build, output, root } from './build.mjs';
await build();
const port = Number(process.env.PORT ?? 4173);
const host = process.env.HOST ?? '127.0.0.1';
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid PORT');
const types = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.woff2':'font/woff2','.txt':'text/plain; charset=utf-8'};
const server = createServer(async (req, res) => {
  if (!['GET','HEAD'].includes(req.method)) {res.writeHead(405, {'Allow':'GET, HEAD'});return res.end();}
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    let file = resolve(output, `.${pathname}`);
    const local = relative(output, file);
    if (local.startsWith(`..${sep}`) || local === '..' || local.startsWith('/') || pathname.includes('\0')) throw new Error('Invalid path');
    let info = await stat(file);
    if (info.isDirectory()) {
      if (!pathname.endsWith('/')) {res.writeHead(308, {'Location':pathname+'/'});return res.end();}
      file=resolve(file,'index.html');
    }
    const bytes=await readFile(file);
    res.writeHead(200, {'Content-Type':types[extname(file)]??'application/octet-stream','Cache-Control':'no-cache','X-Content-Type-Options':'nosniff','Referrer-Policy':'strict-origin-when-cross-origin'});
    res.end(req.method === 'HEAD' ? undefined : bytes);
  } catch {
    res.writeHead(404, {'Content-Type':'text/html; charset=utf-8'});
    const fallback = await readFile(resolve(output,'404.html')).catch(() => Buffer.from('Not found'));
    res.end(req.method === 'HEAD' ? undefined : fallback);
  }
});
server.listen(port, host, () => console.log(`Local: http://${host}:${port}/ru/`));
for (const signal of ['SIGINT','SIGTERM']) process.on(signal,()=>{
  server.close(()=>process.exit(0));
});
