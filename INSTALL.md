# Installing Caramel

Caramel ships three command-line tools:

| Tool | What it does |
|---|---|
| `caramel-run` | Runs a `.crml` program locally or on a remote worker |
| `caramel-lint` | Static analysis: undefined variables, unused parameters, shape problems |
| `crpk-gen` | Generates CRPK/CRRS protocol fixtures |

Only a C++17 compiler is needed to build them. The MLIR compiler (`caramel-opt`)
is a separate, opt-in half — see [For compiler development](#for-compiler-development).

---

## Linux and macOS: one line

```bash
curl -fsSL https://raw.githubusercontent.com/FabianoDicheti/caramel-lang-preview/master/install.sh | sh
```

This downloads the prebuilt binaries for your machine into `~/.local/bin` — no
compiler, no CMake, no sudo. If no prebuilt binary matches your platform it
falls back to building from source.

Then make sure `~/.local/bin` is on your `PATH` (the installer tells you if it
is not):

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

Check it worked:

```bash
caramel-run ~/.local/share/caramel/examples/matmul_relu.crml --flow layer \
  --in x=2x2:1,2,3,4 --in w=2x2:5,6,7,8 --in bias=2x2:0,0,-100,0
# flow 'layer' outputs:
# result : [2x2] = [19, 22, 0, 50]
```

Installer options:

```bash
CARAMEL_PREFIX=/usr/local sh install.sh   # install system-wide (needs sudo)
CARAMEL_VERSION=v0.1.0    sh install.sh   # pin a release
CARAMEL_FROM_SOURCE=1     sh install.sh   # always build from source
```

## Debian / Ubuntu package

```bash
curl -LO https://github.com/FabianoDicheti/caramel-lang-preview/releases/latest/download/caramel-v0.1.0-linux-x86_64.deb
sudo dpkg -i caramel-v0.1.0-linux-x86_64.deb
```

Installs into `/usr/local/bin`, so it is on `PATH` immediately.

## Manual tarball

Download `caramel-<version>-<platform>.tar.gz` from the
[releases page](https://github.com/FabianoDicheti/caramel-lang-preview/releases) and
unpack it anywhere:

```bash
tar -xzf caramel-v0.1.0-linux-x86_64.tar.gz
sudo cp caramel-*/bin/* /usr/local/bin/
```

Platforms published: `linux-x86_64`, `linux-arm64`, `macos-arm64`, `macos-x86_64`.
Verify a download against `SHA256SUMS.txt` from the same release:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing
```

## From source

Needs CMake ≥ 3.20 and a C++17 compiler. Nothing else.

```bash
# Debian/Ubuntu:  sudo apt install -y build-essential cmake git
# Fedora:         sudo dnf install -y gcc-c++ cmake git
# macOS:          xcode-select --install && brew install cmake
git clone https://github.com/FabianoDicheti/caramel-lang-preview.git
cd caramel_lang
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure     # optional
sudo cmake --install build                     # /usr/local by default
```

Install somewhere without sudo:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build -j && cmake --install build
```

The install lays out a normal prefix:

```
<prefix>/bin/{caramel-run,caramel-lint,crpk-gen}
<prefix>/lib/libcaramel_core.a          # link the interpreter into your own C++
<prefix>/include/caramel/**.h
<prefix>/share/caramel/examples/*.crml
```

Build your own package instead of installing:

```bash
cd build && cpack -G TGZ    # tarball
cd build && cpack -G DEB    # .deb (Linux)
```

---

## Editor support (VS Code)

Syntax highlighting, snippets, completions, hover documentation, live linting,
and a run command.

```bash
code --install-extension caramel-vscode-v0.1.0.vsix
```

Download the `.vsix` from the [releases page](https://github.com/FabianoDicheti/caramel-lang-preview/releases),
or build it from the repository:

```bash
cd editors/vscode
npx @vscode/vsce package --no-dependencies --out caramel.vsix
code --install-extension caramel.vsix
```

Highlighting and snippets work on their own; diagnostics and the run command
call `caramel-lint` / `caramel-run`, so install the toolchain too (or point
`caramel.lintPath` / `caramel.runPath` at your build directory).

See [editors/vscode/README.md](editors/vscode/README.md) for every setting.

---

## For compiler development

The MLIR half (`caramel-opt`, the Caramel and FPGA dialects, the lowering
passes) is off by default because it needs a matching LLVM/MLIR 18 install.

```bash
# Ubuntu:
sudo apt install -y llvm-18-dev libmlir-18-dev mlir-18-tools

cmake -S . -B build -G Ninja \
  -DCARAMEL_ENABLE_MLIR=ON \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
cmake --build build
ctest --test-dir build --output-on-failure
```

With `CARAMEL_ENABLE_MLIR=ON` the dialect round-trip, lowering, and FPGA tests
are registered too; without it the suite covers the frontend, interpreter,
quantizer, protocol, dispatch, and both CLIs.

Other build options:

| Option | Default | Meaning |
|---|---|---|
| `CARAMEL_ENABLE_MLIR` | `OFF` | Build the MLIR dialects and `caramel-opt` |
| `CARAMEL_BUILD_TESTS` | `ON` | Build and register the ctest suite |
| `CARAMEL_FIXTURES_DIR` | auto | Where `priority/fixtures` lives; when absent, `test_remote_execute` is skipped |

---

## Publishing a release (maintainer)

```bash
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/release.yml` builds every platform, runs the tests, packages
tarballs, a `.deb` and the `.vsix`, writes `SHA256SUMS.txt`, and creates the
GitHub Release that `install.sh` downloads from.

> **The repository must be public for `install.sh` to work.** `curl` fetching
> the script from `raw.githubusercontent.com` and the release assets are both
> anonymous requests. While the repository is private, share the built tarball
> and `.vsix` directly, or have users clone with credentials and build from
> source.

## Uninstalling

```bash
rm -f ~/.local/bin/{caramel-run,caramel-lint,crpk-gen}
rm -rf ~/.local/share/caramel
sudo dpkg -r caramel                    # if installed from the .deb
code --uninstall-extension caramel.caramel-lang
```
