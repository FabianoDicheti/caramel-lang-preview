# Caramel Language for VS Code

Editor support for [Caramel](https://github.com/FabianoDicheti/caramel-lang-preview),
a quantization-first tensor language with postfix (RPN) computation.

## Features

| Feature | What you get |
|---|---|
| Syntax highlighting | Directives (`crml::`, `calc::`, `in::`), decorators (`@clock`, `@tile`), tensor ops, combinators, stack ops, comments, numbers, strings |
| File icons | `.crml` files get the Caramel icon (enable **Caramel Icons** in the icon theme picker) |
| Snippets | `program`, `flow`, `crml`, `layer`, `in`, `profile`, `clock`, `tile`, plus every tensor generator (`zeros`, `eye`, `tridiag`, `from_spectrum`, ...) |
| Completions | Ops, generators, directives, decorators and combinators, each with inline documentation |
| Hover | Hover any op to see what it computes and its RPN operand order |
| Live diagnostics | Errors and warnings from `caramel-lint`, underlined as you type |
| Run | **Caramel: Run Current File** (`Ctrl+Alt+R` / `Cmd+Alt+R`) executes the file with `caramel-run` |

Snippets and highlighting work with no toolchain installed. Diagnostics and
running need the Caramel CLIs on your `PATH`:

```bash
curl -fsSL https://raw.githubusercontent.com/FabianoDicheti/caramel-lang-preview/master/install.sh | sh
```

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `caramel.runPath` | `caramel-run` | Path to the interpreter CLI |
| `caramel.lintPath` | `caramel-lint` | Path to the linter CLI |
| `caramel.lint.enable` | `true` | Show lint diagnostics |
| `caramel.lint.run` | `onType` | `onType` or `onSave` |
| `caramel.run.flow` | `""` | Flow name passed as `--flow` |
| `caramel.run.inputs` | `[]` | Inputs passed as `--in`, e.g. `["x=2x2:1,2,3,4"]` |

`caramel.runPath` and `caramel.lintPath` accept `${workspaceFolder}` and `~`,
so you can point them at a local build:

```json
{
  "caramel.runPath": "${workspaceFolder}/build/caramel-run",
  "caramel.lintPath": "${workspaceFolder}/build/caramel-lint"
}
```

Example `.vscode/settings.json` for a project whose flow takes two matrices:

```json
{
  "caramel.run.flow": "layer",
  "caramel.run.inputs": ["x=2x2:1,2,3,4", "w=2x2:5,6,7,8", "bias=2x2:0,0,-100,0"]
}
```

## Install

**From a `.vsix`:** download `caramel.vsix` from the
[releases page](https://github.com/FabianoDicheti/caramel-lang-preview/releases), then

```bash
code --install-extension caramel.vsix
```

or in VS Code: Extensions → `...` menu → **Install from VSIX...**

**From source:**

```bash
cd editors/vscode
npx @vscode/vsce package --no-dependencies --out caramel.vsix
code --install-extension caramel.vsix
```

## A first file

Create `hello.crml`, type `program`, and press Tab:

```crml
crml::quantmax=1000;
crml::quantmin=-1000;
crml::quantres=0;

calc::lambda_flow layer(x, w, bias) {
    x w matmul p =
    p bias elemwise_add s =
    s relu result =
} return result;
```

Then press `Ctrl+Alt+R` with the inputs configured as shown above.
