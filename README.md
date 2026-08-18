# Caramel

A quantization-first tensor language. Programs are written in postfix (RPN)
form, values are integers under an explicit quantization range, and the same
program runs on the local interpreter, on a remote worker, or through the MLIR
compiler down to an FPGA dialect.

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

```bash
caramel-run layer.crml --flow layer \
  --in x=2x2:1,2,3,4 --in w=2x2:5,6,7,8 --in bias=2x2:0,0,-100,0
# flow 'layer' outputs:
# result : [2x2] = [19, 22, 0, 50]
```

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/FabianoDicheti/caramel-lang-preview/master/install.sh | sh
```

Prebuilt binaries for Linux and macOS, into `~/.local/bin`, no compiler needed.
Debian package, tarballs, and the source build are in **[INSTALL.md](INSTALL.md)**.

From source it is only CMake (≥ 3.20) plus a C++17 compiler:

```bash
# Debian/Ubuntu: sudo apt install -y build-essential cmake git
cmake -S . -B build && cmake --build build -j && sudo cmake --install build
```

## The toolchain

| Tool | What it does |
|---|---|
| `caramel-run` | Run a `.crml` program locally or on a remote worker |
| `caramel-lint` | Static analysis with error codes and source locations |
| `crpk-gen` | Generate CRPK/CRRS protocol fixtures |
| `caramel-opt` | MLIR compiler driver (opt-in: `-DCARAMEL_ENABLE_MLIR=ON`) |

## Editor support

The VS Code extension in [`editors/vscode/`](editors/vscode/) gives you syntax
highlighting, 28 snippets, completions and hover docs for every op, live
`caramel-lint` diagnostics, and `Ctrl+Alt+R` to run the current file. See
[VSCODE_SETUP.md](VSCODE_SETUP.md).

## Learn the language

| Document | For |
|---|---|
| [docs/LANGUAGE_GUIDE.md](docs/LANGUAGE_GUIDE.md) | The language, from RPN basics to devices |
| [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) | Running programs and the CLI |
| [examples/](examples/) | Runnable `.crml` programs |

## Status

This is a preview drop of the Caramel toolchain, published so it can be
installed and tried. No license has been chosen yet, so the code is under
default copyright: you may read and build it, but redistribution rights are not
granted.
