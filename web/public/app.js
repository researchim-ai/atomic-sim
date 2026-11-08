/* Progressive enhancement: navigation and all project content also work without JS. */
(() => {
  const toast = document.querySelector('#toast');
  let toastTimeout;
  const announce = (message) => {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('visible');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => toast.classList.remove('visible'), 2600);
  };
  const figure = document.querySelector('.reactor-figure');
  const scenarioData = document.querySelector('#scenario-data');
  if (figure && scenarioData) {
    const scenarios = JSON.parse(scenarioData.textContent);
    document.querySelectorAll('[data-scenario-button]').forEach(button => {
      button.addEventListener('click', () => {
        const scenario = scenarios.find(s => s.id === button.dataset.scenarioButton);
        if (!scenario) return;
        figure.dataset.scenario = scenario.id;
        document.querySelectorAll('[data-scenario-button]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
        document.querySelector('#scenario-description').textContent = scenario.text;
      });
    });
  }
  const motion = document.querySelector('.motion-toggle');
  if (motion && figure) motion.addEventListener('click', () => {
    const paused = motion.getAttribute('aria-pressed') !== 'true';
    motion.setAttribute('aria-pressed', String(paused));
    figure.dataset.paused = String(paused);
    motion.querySelector('.motion-label').textContent = paused ? motion.dataset.resumeLabel : motion.dataset.pauseLabel;
  });
  const tabs = [...document.querySelectorAll('[data-code-tab]')];
  function selectTab(tab, focus = false) {
    tabs.forEach(t => {
      const selected = t === tab;
      t.setAttribute('aria-selected', String(selected));
      t.tabIndex = selected ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !selected;
    });
    document.querySelector('.copy-code').dataset.copy = `snippet-${tab.dataset.codeTab}`;
    document.querySelector('.window-top>span').textContent = {
      python: 'experiment.py', gym: 'environment.py', llm: 'local_tools.py',
    }[tab.dataset.codeTab];
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectTab(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        selectTab(tabs[next], true);
      }
    });
  });
  document.querySelectorAll('[data-copy]').forEach(button => {
    button.addEventListener('click', async () => {
      const source = document.getElementById(button.dataset.copy);
      if (!source) return;
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(source.textContent);
        } else {
          const temporary = document.createElement('textarea');
          temporary.value = source.textContent;
          temporary.setAttribute('readonly', '');
          temporary.style.cssText = 'position:fixed;top:0;left:-9999px';
          document.body.append(temporary);
          try {
            temporary.select();
            if (!document.execCommand('copy')) throw new Error('Clipboard denied');
          } finally {
            temporary.remove();
            button.focus();
          }
        }
        announce(button.dataset.copied);
      } catch {
        announce(button.dataset.error);
      }
    });
  });
  const toggle = document.querySelector('.menu-toggle');
  const menu = document.querySelector('#mobile-nav');
  function closeMenu(returnFocus = false) {
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-label', toggle.dataset.openLabel);
    menu.classList.remove('open');
    if (returnFocus) toggle.focus();
  }
  if (toggle && menu) {
    toggle.addEventListener('click', () => {
      const opened = toggle.getAttribute('aria-expanded') !== 'true';
      toggle.setAttribute('aria-expanded', String(opened));
      toggle.setAttribute('aria-label', opened ? toggle.dataset.closeLabel : toggle.dataset.openLabel);
      menu.classList.toggle('open', opened);
    });
    menu.querySelectorAll('a').forEach(a => a.addEventListener('click', () => closeMenu()));
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && menu.classList.contains('open')) closeMenu(true);
    });
    document.addEventListener('click', e => {
      if (!menu.contains(e.target) && !toggle.contains(e.target)) closeMenu();
    });
    const desktop = matchMedia('(min-width: 761px)');
    desktop.addEventListener('change', () => { if (desktop.matches) closeMenu(); });
  }
  document.querySelectorAll('.language-switch a').forEach(a => {
    a.addEventListener('click', () => {
      if (location.hash) a.hash = location.hash;
    });
  });
})();
