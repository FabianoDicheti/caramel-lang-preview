# VS Code Setup for Caramel

The extension lives in [`editors/vscode/`](editors/vscode/). It provides syntax
highlighting, file icons, snippets, completions, hover documentation, live
`caramel-lint` diagnostics, and a run command.

## Install

### From a packaged `.vsix` (what you share with others)

```bash
code --install-extension caramel-vscode-v0.1.0.vsix
```

Download it from the
[releases page](https://github.com/FabianoDicheti/caramel-lang-preview/releases), or in
VS Code: Extensions → `...` → **Install from VSIX...**

### Build the `.vsix` from this repository

```bash
cd editors/vscode
npx @vscode/vsce package --no-dependencies --out caramel.vsix
code --install-extension caramel.vsix
```

### Develop against it

Open `editors/vscode/` in VS Code and press `F5` to launch an Extension
Development Host, or symlink it:

```bash
ln -s "$PWD/editors/vscode" ~/.vscode/extensions/caramel-lang
```

Then reload: `Cmd/Ctrl+Shift+P` → "Developer: Reload Window".

## Activate the icon theme

`Cmd/Ctrl+Shift+P` → "Preferences: File Icon Theme" → **Caramel Icons**.

## Features

| Feature | Notes |
|---|---|
| Syntax highlighting | `syntaxes/caramel.tmLanguage.json` — directives, namespaces, decorators, ops, combinators, stack ops, numbers, `#` comments |
| File icons | `.crml` files show the Caramel logo |
| Snippets | `snippets/caramel.json` — type `program`, `flow`, `layer`, `crml`, `zeros`, `eye`… and press Tab |
| Completions + hover | Every op, generator, directive, decorator and combinator, with inline docs |
| Diagnostics | Runs `caramel-lint` as you type; underlines errors/warnings with their codes |
| Run | **Caramel: Run Current File** (`Ctrl+Alt+R` / `Cmd+Alt+R`) runs `caramel-run` and prints to the Caramel output channel |

Diagnostics and running need `caramel-lint` / `caramel-run` on `PATH` — see
[INSTALL.md](INSTALL.md) — or point `caramel.lintPath` / `caramel.runPath` at
your build directory:

```json
{
  "caramel.runPath": "${workspaceFolder}/build/caramel-run",
  "caramel.lintPath": "${workspaceFolder}/build/caramel-lint"
}
```

## Verify

Open any `.crml` file in `examples/`. You should see the Caramel icon on the
tab, colored syntax, and — if the toolchain is installed — squiggles on a
deliberate typo.
