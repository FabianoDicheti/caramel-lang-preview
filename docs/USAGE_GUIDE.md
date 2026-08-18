# Caramel Usage Guide

This guide takes a new engineer from a clean machine to a repeatable daily
workflow: build, test, lint, run, benchmark, and diagnose Caramel programs.

## 1. Prerequisites and supported setup

**Learning objectives**

- Install the project toolchain.
- Understand which platform path is authoritative.
- Confirm CMake can find LLVM and MLIR.

The project uses C++17, CMake 3.20 or newer, Ninja, and LLVM/MLIR 18. The
repository-verified environment is Ubuntu 24.04 with packaged LLVM/MLIR 18.
The macOS path is useful but best-effort. Consult the official
[MLIR getting-started guide](https://mlir.llvm.org/getting_started/) for upstream
toolchain context and the current [Homebrew `llvm@18` formula](https://formulae.brew.sh/formula/llvm%4018)
before relying on package availability.

### Ubuntu 24.04: verified path

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build \
  llvm-18-dev libmlir-18-dev mlir-18-tools clang-18
```

Configure, build, and test:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
cmake --build build
ctest --test-dir build --output-on-failure
```

If Ubuntu cannot find the versioned packages, verify that the LLVM 18 package
repository appropriate for the machine is enabled rather than silently building
against a different MLIR major version.

### macOS: best-effort Homebrew path

```bash
brew install cmake ninja llvm@18
```

Use Homebrew's keg-specific CMake packages:

```bash
LLVM18_PREFIX="$(brew --prefix llvm@18)"
cmake -S . -B build -G Ninja \
  -DLLVM_DIR="$LLVM18_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM18_PREFIX/lib/cmake/mlir" \
  -DCMAKE_C_COMPILER="$LLVM18_PREFIX/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM18_PREFIX/bin/clang++"
cmake --build build
ctest --test-dir build --output-on-failure
```

`llvm@18` is keg-only, so do not assume `/usr/local/bin` or `/opt/homebrew/bin`
contains its tools. Use `brew --prefix llvm@18` as shown. If Homebrew's package
layout lacks `lib/cmake/mlir`, stop and report the platform/toolchain mismatch;
do not substitute an unrelated LLVM major version.

**Knowledge check:** Why are both `LLVM_DIR` and `MLIR_DIR` supplied?  
They point CMake at the installed configuration packages used by this out-of-tree
MLIR project.

## 2. Know the build outputs

**Learning objectives**

- Choose the correct executable.
- Build a single target during focused development.

| Target | Use |
|---|---|
| `caramel-run` | Parse and execute `.crml` locally or remotely. |
| `caramel-lint` | Report static source diagnostics. |
| `caramel-opt` | Parse Caramel MLIR and run compiler passes. |
| `caramel-mlir-smoke` | Check basic MLIR integration. |
| `crpk-gen` | Generate protocol fixtures. |

Build only one target when iterating:

```bash
cmake --build build --target caramel-run
```

For the full target and artifact map, see
[Build and artifacts](../INSTALL.md).

## 3. Editor setup

**Learning objectives**

- Associate `.crml` files with Caramel.
- Know the boundary of editor support.

Follow `VSCODE_SETUP.md` to load the repository's VS Code extension. It provides
syntax highlighting, comments, bracket behavior, and icons. It does not provide
a language server, completion, or live compiler diagnostics. Run the actual
linter and parser before trusting syntax color alone.

## 4. Your first run

**Learning objectives**

- Supply scalar and tensor parameters.
- Select a flow and interpret output formatting.

Run the checked-in matrix example:

```bash
./build/caramel-run examples/matmul_relu.crml \
  --flow layer \
  --in x=2x2:1,2,3,4 \
  --in w=2x2:5,6,7,8 \
  --in bias=2x2:0,0,-100,0
```

Expected output includes:

```text
flow 'layer' outputs:
result : [2x2] = [19, 22, 0, 50]
```

Input forms:

| Form | Meaning |
|---|---|
| `--in count=5` | Scalar integer input. |
| `--in x=2x3:1,2,3,4,5,6` | Rank-2 tensor, row-major. |

The number of tensor values must equal the product of its dimensions. Repeating
an input name uses the last value. Every selected flow parameter needs a value
unless the script supplies a compatible input directive.

## 5. Lint before running

**Learning objectives**

- Separate parse errors from lint diagnostics.
- Use diagnostic codes to guide fixes.

```bash
./build/caramel-lint examples/matmul_relu.crml
```

The linter detects issues such as duplicate callable definitions/targets,
undefined variables, reassignment, unused parameters/variables, duplicate object
keys, and malformed device declarations. A parse failure occurs earlier and
prevents AST linting.

A good loop is:

```bash
./build/caramel-lint path/to/program.crml
./build/caramel-run path/to/program.crml --flow name --in ...
```

## 6. Selecting flows and diagnosing failures

**Learning objectives**

- Select one flow from a multi-flow file.
- Classify common failure modes.

Use `--flow NAME` when a file contains multiple flows. Typical failures:

| Message class | Likely cause | Next action |
|---|---|---|
| Cannot open file | Wrong path or working directory | Resolve the path from repository root. |
| Parse error with line/column | Invalid grammar or insufficient RPN operands | Inspect the reported token and simulate the stack. |
| Missing input | Flow parameter lacks `--in` data | Add the named input. |
| Bad `--in` | Shape/value count mismatch or malformed scalar | Recalculate dimensions and use comma-separated integers. |
| Unsupported op | Parsed operation has no selected backend evaluator | Check the language capability status. |
| Incompatible shapes | Elementwise or matrix dimension mismatch | Trace shapes at each binding. |

## 7. Tests and focused development

**Learning objectives**

- Run the full regression suite.
- Run a focused CTest target during iteration.

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R test_parser --output-on-failure
ctest --test-dir build -R caramel_run_cli --output-on-failure
```

Use Testing to map a change to the relevant suite. Run the full
suite before handing work to another engineer.

The objective harness provides broader project gates:

```bash
bash scripts/verify_objectives.sh
```

Read `tests/verify/README.md` first; these checks include numerical and performance
expectations beyond ordinary unit tests.

## 8. MLIR usage

**Learning objectives**

- Run a compiler pass on a checked-in MLIR fixture.
- Distinguish `.crml` source execution from MLIR transformations.

`.crml` programs run through `caramel-run`. MLIR text runs through `caramel-opt`.

```bash
./build/caramel-opt \
  --lower-caramel-to-linalg \
  tests/mlir/lower_to_linalg.mlir
```

Other registered project pass arguments include `--caramel-to-affine`,
`--lower-caramel-to-fpga`, and `--profile-matmul`. Use the matching test fixture
as a known-good starting point.

## 9. Device discovery and remote execution

**Learning objectives**

- Discover workers.
- Execute explicitly against a worker.
- Compare remote results with the local reference.

Discover by broadcast and/or explicit probes:

```bash
./build/caramel-run devices
./build/caramel-run devices --probe HOST:PORT --timeout 1000 --no-broadcast
```

Run one flow remotely:

```bash
./build/caramel-run program.crml \
  --flow flow_name \
  --in x=... \
  --device http://HOST:PORT \
  --device-user USER \
  --device-pass env:CARAMEL_DEVICE_PASSWORD
```

A pre-issued token can replace user/password authentication:

```bash
--device-token env:CARAMEL_DEVICE_TOKEN
```

Do not combine token authentication with user/password authentication. Credential
flags require `--device`. Use `--verify` to run locally and remotely and compare
decoded outputs:

```bash
./build/caramel-run program.crml ... \
  --device http://HOST:PORT \
  --device-token env:CARAMEL_DEVICE_TOKEN \
  --verify
```

Script-driven device declarations can activate dependency-ordered dispatch when
no explicit `--device` overrides them. Remote work also requires protocol and
operation compatibility with the worker. The
remote worker guide provides progressive,
self-contained examples for this mode.

## 10. Benchmarks

**Learning objectives**

- Run a representative workload.
- Separate correctness from timing.

The benchmark harness and its Python baselines are not part of this preview
drop. The timing gates that do ship run under CTest:

```bash
ctest --test-dir build -R "bench_|verify_" --output-on-failure
```

Do not accept a faster result that fails output validation. Benchmarks are
measurements; CTest and verification suites are correctness gates.

## 11. Daily engineering workflow

1. Pull or inspect the intended change and read the relevant technical page.
2. Configure once; build the smallest affected target.
3. Add or adjust a focused test before changing semantics.
4. Run `caramel-lint` on edited `.crml` files.
5. Run the focused test, then the full CTest suite.
6. Run objective/benchmark checks when touching performance, quantization, or IR.
7. Update examples and documentation when behavior changes.
8. Review `git diff` and preserve unrelated working-tree changes.

## 12. Progressive labs

### Lab 1: environment proof

Configure the repository, build `caramel-run`, and record the LLVM version printed
by CMake.

### Lab 2: input validation

Run `examples/matmul_relu.crml` successfully, then deliberately supply only three
values for a `2x2` input. Explain the failure.

### Lab 3: lint repair

Run the linter on `tests/cli/caramel_lint_invalid.crml`. Identify one error and
one warning, then describe the source change that would address each.

### Lab 4: focused regression

Run only parser tests, then run the complete suite. Explain when each command is
appropriate.

### Lab 5: compiler pass

Run the Linalg lowering fixture and identify the dialect of the resulting core
operation.

## 13. Solutions

### Lab 1

Use the platform-specific configure command from section 1, followed by:

```bash
cmake --build build --target caramel-run
```

CMake should report LLVM/MLIR major version 18.

### Lab 2

The successful command is shown in section 4. A `2x2` tensor requires four
row-major values, so `2x2:1,2,3` is rejected as a bad `--in`.

### Lab 3

Use:

```bash
./build/caramel-lint tests/cli/caramel_lint_invalid.crml
```

Resolve undefined names by defining/passing them before use; resolve unused
bindings by using them or removing them. Preserve warnings that intentionally
teach a lint rule only in test fixtures.

### Lab 4

```bash
ctest --test-dir build -R test_parser --output-on-failure
ctest --test-dir build --output-on-failure
```

The first shortens the edit/debug loop; the second detects cross-subsystem
regressions before handoff.

### Lab 5

```bash
./build/caramel-opt --lower-caramel-to-linalg \
  tests/mlir/lower_to_linalg.mlir
```

The Caramel matrix operation is lowered into the standard Linalg path. Compare
the output with the `CHECK` expectations in the fixture.
