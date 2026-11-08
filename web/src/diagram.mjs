export function reactorDiagram(t) {
  const rods = Array.from({length: 9}, (_, i) => {
    const x = 214 + i * 10;
    const y = 245 + Math.abs(i - 4) * 3;
    return `<path class="fuel-rod" d="M${x} ${y}v91"/><ellipse cx="${x}" cy="${y}" rx="3.3" ry="2.1" fill="#f09562"/>`;
  }).join('');
  const controls = [235, 255, 275].map(x => `<g class="control-rod"><path d="M${x} 84v167"/><rect x="${x-5}" y="78" width="10" height="9" rx="2"/></g>`).join('');
  return `<svg class="reactor-diagram" viewBox="0 0 600 510" role="img" aria-label="${t.figureAlt}">
    <defs>
      <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#d6d9ce" stroke-width=".65"/></pattern>
      <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(35)"><path d="M0 0V7" stroke="#526258" stroke-width="1" opacity=".3"/></pattern>
      <linearGradient id="vessel" x1="0" x2="1"><stop stop-color="#e1e5d9"/><stop offset=".4" stop-color="#f7f7ed"/><stop offset="1" stop-color="#cad2c4"/></linearGradient>
      <linearGradient id="core" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#f6ad79" stop-opacity=".1"/><stop offset="1" stop-color="#ed7a41" stop-opacity=".4"/></linearGradient>
      <marker id="orange-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 1l9 4-9 4" fill="none" stroke="#d97642" stroke-width="1.5"/></marker>
      <marker id="green-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 1l9 4-9 4" fill="none" stroke="#517c61" stroke-width="1.5"/></marker>
    </defs>
    <rect x="0" y="0" width="600" height="510" fill="url(#grid)" opacity=".65"/>
    <g class="draft-lines" fill="none" stroke="#9ca99a" stroke-width=".7">
      <path d="M25 18v12M19 24h12M575 18v12M569 24h12M25 480v12M19 486h12M575 480v12M569 486h12"/>
      <path d="M255 40v404" stroke-dasharray="5 7"/><path d="M42 412h512" stroke-dasharray="5 7"/>
      <path d="M174 153H157v220h17M152 153h10M152 373h10"/>
    </g>
    <g class="secondary" fill="none" stroke="#79958b" stroke-width="3">
      <path d="M485 176v-63q0-12 12-12h49" stroke-dasharray="5 5"/>
      <path d="M546 365h-35q-13 0-13-13v-24"/>
    </g>
    <g class="pipe-back" fill="none" stroke-linejoin="round"><path d="M332 205h69q17 0 17 17v65q0 17 17 17h23"/><path d="M463 341v53q0 18-18 18H99v-75q0-17 17-17h70"/></g>
    <path class="hot-pipe" d="M332 205h69q17 0 17 17v65q0 17 17 17h23"/>
    <path class="cool-pipe" d="M463 341v53q0 18-18 18H99v-75q0-17 17-17h70"/>
    <path class="flow-dashes hot-flow" d="M332 205h69q17 0 17 17v65q0 17 17 17h23"/>
    <path class="flow-dashes cool-flow" d="M463 341v53q0 18-18 18H99v-75q0-17 17-17h70"/>
    <g class="steam-generator" stroke="#344d40" stroke-width="1.5">
      <path d="M446 185v150c0 22 55 22 55 0V185" fill="url(#vessel)"/>
      <ellipse cx="473.5" cy="185" rx="27.5" ry="10" fill="#eef0e4"/>
      <path d="M456 208q35 6 0 19q35 6 0 19q35 6 0 19q35 6 0 19q35 6 0 19q35 6 0 19" fill="none" stroke="#c57647" stroke-width="3"/>
      <path d="M451 195v133M493 195v133" opacity=".3"/>
    </g>
    <g class="vessel-support" stroke="#344d40" stroke-width="1.4">
      <path d="M203 369v32l17 8v-35M293 373v28l17 8v-42" fill="url(#hatch)"/>
      <path d="M187 410h43M283 410h38"/>
    </g>
    <g stroke="#344d40" stroke-width="1.6">
      <path d="M182 159v197c0 42 151 42 151 0V159" fill="url(#vessel)"/>
      <ellipse cx="257.5" cy="159" rx="75.5" ry="26" fill="#d4dccd"/>
      <ellipse cx="257.5" cy="159" rx="59" ry="19" fill="#f5f4e8"/>
      <path d="M198 164v181c0 31 120 31 120 0V164" fill="none" stroke-dasharray="4 4" opacity=".5"/>
      <path class="core-glow" d="M207 225q47-20 96 0v114q-47 25-96 0z" fill="url(#core)" stroke="none"/>
      <path d="M207 339q47 27 96 0v13q-47 27-96 0z" fill="#80937c"/>
      <g stroke="#d47b45" stroke-width="6" stroke-linecap="round">${rods}</g>
      <path d="M182 159c0-26 25-44 50-47h50c27 4 51 22 51 47" fill="url(#vessel)"/>
      <path d="M203 130q53 27 110 0" fill="none" stroke="#8d9d89"/>
      <path d="M229 115V99q28-12 56 0v16" fill="#dee3d5"/>
      <ellipse cx="257" cy="99" rx="28" ry="8" fill="#f5f4e8"/>
      ${controls}
      <path d="M194 195v112" stroke="#f9f9f0" stroke-width="4" opacity=".65"/>
    </g>
    <g class="pump-body" transform="translate(99 412)" stroke="#344d40" stroke-width="1.6">
      <circle r="28" fill="#f3f3e8"/><circle r="20" fill="#dce5d5"/>
      <g class="pump-rotor"><path d="M0-4c-20-19 12-20 5-1M4 1c26-4 8 24-2 5M-2 4c-8 24-22-3-3-6" fill="#6e8e74" stroke="none"/></g>
      <circle r="4" fill="#344d40"/>
    </g>
    <g fill="none" stroke="#6c7e6e" stroke-width=".9">
      <path d="M64 194h48l65 48"/><circle cx="177" cy="242" r="2.5" fill="#f3f3e8"/>
      <path d="M473 160v-22h-52"/><path d="M99 446v13"/>
      <path d="M335 273h32l23 26"/><circle cx="335" cy="273" r="2.5" fill="#f3f3e8"/>
    </g>
    <g class="diagram-labels" fill="#344d40" font-family="'Lab Mono', monospace" font-size="10" letter-spacing="1">
      <text x="49" y="180">01 / ${t.reactor}</text>
      <text x="389" y="128">02 / ${t.generator}</text>
      <text x="62" y="478">03 / ${t.pump}</text>
      <text x="522" y="89" font-size="8">${t.steam} ↗</text>
      <text x="514" y="385" font-size="8">${t.feed}</text>
      <text x="368" y="316" fill="#c66e3c" font-size="8">${t.heat}</text>
      <text x="261" y="444" fill="#517c61" font-size="8">${t.flow} ←</text>
      <text x="294" y="78" font-size="8">${t.control}</text>
    </g>
    <g class="core-particles" fill="#e88249"><circle cx="227" cy="227" r="2"/><circle cx="290" cy="240" r="2.5"/><circle cx="251" cy="311" r="2"/></g>
  </svg>`;
}
