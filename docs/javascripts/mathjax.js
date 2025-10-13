window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['noscript', 'style', 'textarea', 'pre', 'code'],
    renderActions: {
      addMenu: []
    }
  }
};

// Re-typeset on navigation (Material for MkDocs)
document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});

// Re-typeset before print-site PDF generation
window.addEventListener('beforeprint', () => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
