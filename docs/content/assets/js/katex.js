// Render the math of pymdownx.arithmatex (generic mode: \( \) and \[ \]) with
// KaTeX, on every page load, including instant navigation.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  })
})
