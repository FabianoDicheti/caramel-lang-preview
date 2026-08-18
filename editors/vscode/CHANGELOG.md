# Changelog

## 0.1.0

- Syntax highlighting for `.crml` (directives, namespaces, decorators, ops,
  combinators, stack operations, numbers, comments).
- File icon theme for `.crml`.
- 28 snippets: program skeleton, flows, directives, tensor generators,
  decorators, and the canonical matmul/bias/relu layer.
- Completions and hover documentation for ops, generators, directives,
  decorators, and combinators.
- Live diagnostics from `caramel-lint`, as you type or on save.
- `Caramel: Run Current File` (Ctrl/Cmd+Alt+R) runs the buffer through
  `caramel-run` and streams the output into the Caramel output channel.
