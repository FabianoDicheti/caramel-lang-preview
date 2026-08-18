# Objectives Verification Harness

Verifies `caramel_lang` against the project's **original objectives** (not just unit
correctness) and answers *"how near are we to the original vision?"*.

## Run

```
bash scripts/verify_objectives.sh
cat work/reports/OBJECTIVES_VERIFICATION.md   # the scorecard / answer
```

The script: builds, runs the regression gate (`ctest`), runs the objective tests
below, runs an end-to-end compile-path check (`caramel-run` + `caramel-opt`), and
**auto-generates** the scorecard report. Each test prints machine-parseable
`RESULT <name> <PASS|FAIL|MEASURE|NOTE> ...` lines that the report aggregates.

## Tests → objective

| Test | Objective (source) |
|------|--------------------|
| `bench_compile` | compile < 1s / 1000 LOC (README.md:441) |
| `bench_parse_scaling` | O(n) parsing (LANGUAGE_SPEC_RPN.md:64) |
| `verify_quant_accuracy` | quantization < 5% accuracy loss (README.md:443) |
| `verify_sim_precision` | simulation matches FPGA integer/saturating precision (ROADMAP M2) |
| `verify_integer_only` | integer-only runtime / compile-time quantization (spec:36,199) |
| `verify_zero_overhead` | object params → positional; stack ops free (spec:40,736) |
| `verify_contract` | binary IR 64-bit + CRML + contract op coverage (INTEGRATION_CONTRACT.md) |

The tests are also registered as ctest targets, so `ctest` runs them too. The
generated report classifies every original objective as **Met / Partial / Gap /
Cross-workstream** and is the source of truth for the gap analysis.
