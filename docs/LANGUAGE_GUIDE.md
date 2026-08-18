# Caramel Language Guide

Welcome to Caramel. This guide teaches the language from the perspective of an
engineer who needs to read, write, and reason about `.crml` programs. It covers
the complete surface grammar while making backend support visible.

Use this guide together with the [usage guide](USAGE_GUIDE.md) when
running examples and the compiler development guide
when changing the implementation.

## 1. How to read feature status

Caramel's grammar is broader than any single execution backend. Each feature in
this guide uses one or more status labels:

| Label | Meaning |
|---|---|
| **Local runtime** | The current `caramel-run` interpreter executes it. |
| **Parser** | The lexer/parser represent it in the AST, but local execution may not exist. |
| **MLIR** | The Caramel MLIR dialect or passes represent/transform it. |
| **Device** | It participates in remote/FPGA orchestration and requires a compatible worker. |

“Parser” is not the same as “runnable.” When learning, begin with examples marked
**Local runtime**, then use broader features when working on their backend.
Short flow-only snippets below assume the required quantization header shown in
the complete examples.

## 2. The central idea: postfix computation

**Learning objectives**

- Read Reverse Polish Notation (RPN) from left to right.
- Predict the operand stack after every token.
- Recognize the assignment target at the end of an expression.

Most languages write addition as `a + b`. Caramel writes:

```crml
a b add result =
```

Read it as: push `a`, push `b`, apply `add`, bind the result to `result`.

| Token | Stack after the token |
|---|---|
| `a` | `a` |
| `b` | `a, b` |
| `add` | `add(a, b)` |
| `result =` | empty; the value is bound to `result` |

There is no operator precedence. This:

```crml
a b add c mul result =
```

means `(a + b) * c`. The stack makes grouping explicit:

1. `a b add` produces one value.
2. `c` pushes another value.
3. `mul` consumes both.

The current local runtime supports integer scalar/tensor `add`, `sub`, `mul`,
`div`, their `elemwise_*` forms, `relu`, `scalar_mul`, `matmul`, `transpose`,
and the four `tensor_*` reductions. Division by zero currently produces zero,
and tensor means use integer division.

**Knowledge check:** What does `a b mul c add out =` compute?  
Answer after reasoning with the stack: `(a * b) + c`.

## 3. A complete minimal program

**Learning objectives**

- Recognize required program directives.
- Define a flow with parameters and outputs.
- Follow intermediate values.

```crml
crml::quantmax=1000;
crml::quantmin=-1000;
crml::quantres=0;

calc::lambda_flow add_and_scale(a, b, scale) {
    a b add sum =
    sum scale scalar_mul result =
} return result;
```

Status: **Local runtime**.

Every program begins with quantization declarations. A named `lambda_flow`
declares its inputs, body, and returned bindings. Statements end with a newline
or semicolon. Comments start with `#`.

Run it after building:

```bash
./build/caramel-run program.crml \
  --flow add_and_scale \
  --in a=3 --in b=4 --in scale=2
```

Expected value: `14`.

## 4. Directives and program configuration

**Learning objectives**

- Separate compile-time directives from runtime values.
- Recognize quantization, autodiff, checkpoint, and routing configuration.

Directives use the `crml::key=value;` or specialized call/object form.

| Family | Examples | Status |
|---|---|---|
| Quantization | `quantmax`, `quantmin`, `quantres`, `quantrange`, `decimal_places`, forward/gradient variants | Parser; the three basic directives feed runtime/dispatch quantization |
| Autodiff | `enable_autodiff`, `autodiff`, `gradient_checkpointing` | Parser |
| Checkpointing | `checkpoint_every` | Parser |
| Routing | `routing_policy { ... }` | Parser/device-oriented |
| Custom backward | `register_backward(forward, backward);` | Parser |

The common header is:

```crml
crml::quantmax=1000;
crml::quantmin=-1000;
crml::quantres=0;
```

These are configuration declarations; they do not push values onto the RPN
stack. Keep them together at the top of a file.

## 5. Values, literals, references, and types

**Learning objectives**

- Write scalar, string, and tensor literals.
- Understand shape requirements.
- Recognize namespaced, indexed, and property references.

### Literals

```crml
42
-7
3.25
"double-quoted text"
'single-quoted text'
[[1, 2], [3, 4]]
```

Tensor literals must be rectangular. `[[1, 2], [3]]` is rejected because its
rows have different lengths. The local interpreter stores integer data; floating
values belong to the wider syntax/quantization model and are not general-purpose
local floating-point runtime values.

### References

| Form | Example | Purpose/status |
|---|---|---|
| Variable | `weights` | Local binding; local runtime |
| Qualified | `port::0`, `receiver::matrix_a`, `memory::W1`, `responser::output` | External/device binding; parser/device |
| Indexed | `A[0]`, `A[0:32]`, `A[1][2]` | Indexed/sliced value; parser |
| Property | `decomp.eigenvalues` | Structured-result property; parser |

### Type annotations

An assignment can place a type annotation before `=`:

```crml
a b matmul product oc::tensor =
```

Namespaces are `oc`, `ot`, and `os`; built-in type names include `number`,
`tensor`, `string`, `text`, and `dictionary`. Modifiers such as `trunc` follow
another `::`. Type annotations are represented in the AST; do not assume every
annotation changes local interpreter behavior.

## 6. Operations

**Learning objectives**

- Select operations by arity.
- Distinguish registry support from backend execution support.

| Group | Examples | Current status |
|---|---|---|
| Arithmetic | `add`, `sub`, `mul`, `div` | Local runtime |
| Elementwise | `elemwise_add`, `elemwise_sub`, `elemwise_mul`, `elemwise_div` | Local runtime |
| Activation | `relu` | Local runtime |
| Matrix/shape | `matmul`, `transpose`, `scalar_mul` | Local runtime |
| Reduction | `tensor_sum`, `tensor_mean`, `tensor_max`, `tensor_min` | Local runtime |
| Comparison/logic | `>`, `<`, `==`, `and`, `or`, `not`, `xor` | Parser; selected test/evaluator paths exist, so verify the chosen backend |
| Neural/tensor | convolution, pooling, normalization, reshape, concat, split | Parser/operation registry; backend-dependent |
| Advanced math | decompositions, FFT, attention, selective scan, S4 | Parser/operation registry; backend-dependent |
| Host operations | loss, gradient clipping, `exp`, `sqrt`, embedding, text/time | Parser/operation registry; backend-dependent |

Operation arity matters. `matmul` consumes two values; `relu` consumes one.
Object parameters carry named configuration:

```crml
x kernel conv2d {stride: 1, padding: 0} result =
```

Status: **Parser/operation registry**. It is a valid illustration of configured
operation syntax, not a promise of local convolution execution.

## 7. Tensors and a linear layer

**Learning objectives**

- Track matrix dimensions through a pipeline.
- Understand row-major CLI inputs.

```crml
crml::quantmax=1000;
crml::quantmin=-1000;
crml::quantres=0;

calc::lambda_flow layer(x, w, bias) {
    x w matmul product =
    product bias elemwise_add shifted =
    shifted relu result =
} return result;
```

Status: **Local runtime**. For `x` and `w` shaped `2x2`:

```bash
./build/caramel-run examples/matmul_relu.crml \
  --flow layer \
  --in x=2x2:1,2,3,4 \
  --in w=2x2:5,6,7,8 \
  --in bias=2x2:0,0,-100,0
```

The values move through these shapes:

| Binding | Computation | Shape/value |
|---|---|---|
| `product` | `matmul(x, w)` | `[[19,22],[43,50]]` |
| `shifted` | `product + bias` | `[[19,22],[-57,50]]` |
| `result` | `relu(shifted)` | `[[19,22],[0,50]]` |

## 8. Stack operations

**Learning objectives**

- Manipulate stack values without introducing bindings.
- Use stack operations only when they clarify data reuse.

Caramel defines `dup`, `swap`, `drop`, `over`, `rot`, and `-rot`.

```crml
x dup matmul x_squared =
```

`dup` turns stack `[x]` into `[x, x]`; `matmul` then consumes both. Stack
operations are parsed into dedicated AST nodes. Prefer named intermediate
bindings when a longer stack sequence would be difficult to review.

## 9. Flows, symphonies, calculators, and clocks

**Learning objectives**

- Distinguish the three top-level calculator forms.
- Read clock-partitioned work.

| Construct | Purpose | Status |
|---|---|---|
| `calc::lambda_flow` | Named inputs, computations, and returned values | Parser; local runtime/dispatch uses flows |
| `calc::lambda_symphony` | Higher-level named computation grouping | Parser |
| `calc::lambda_calculator(channels=N)` | Calculator body with optional channel count | Parser |

Clock sections make logical stages explicit:

```crml
calc::lambda_flow staged(a, b, c, d) {
    @clock(1):
    a b add left =
    c d mul right =
    @clock(2):
    left right add result =
} return result;
```

Status: **Local runtime** for this operation set. Clock 2 depends on values
created in clock 1. Clocks describe phases in the AST/dataflow model; they are not
a license to ignore data dependencies.

## 10. Decorators and object parameters

**Learning objectives**

- Attach metadata to supported targets.
- Keep compile-time options separate from runtime operands.

Built-in decorators include `quantize`, `tile`, `device`, `clock`, `autodiff`,
and `checkpoint`.

```crml
@autodiff
calc::lambda_flow network(x, weights) {
    @tile {rows: 2, cols: 2}
    x weights matmul output =
} return output;
```

Status: **Parser**. Decorator schemas validate allowed names, target kinds, and
whether arguments are required. Backend behavior depends on the decorator.

Object parameters use `{key: value}` and may contain literals, identifiers,
tuples, tensors, and lists. Duplicate keys produce lint warnings.

## 11. Lambdas and combinators

**Learning objectives**

- Recognize arrow and block lambda syntax.
- Understand that combinators are named lambda-calculus primitives.

```crml
x => f(x)
lambda x => x end
```

Status: **Parser/lambda evaluator**, with integration depending on the enclosing
construct. The grammar also reserves bird combinators such as `identity`,
`kestrel`, `kite`, `compose`, `bluebird`, and `ycombinator`. Use the lambda tests
as the executable source of truth before relying on a combinator in a CLI flow.

## 12. Printing, memory, devices, and backpropagation

**Learning objectives**

- Recognize side-effecting and device-oriented statements.
- Avoid confusing printed `return(...)` with a flow return clause.

| Form | Meaning/status |
|---|---|
| `return(value);` | Print statement; parser |
| `} return result;` | Flow output clause |
| `value memory::write` | Memory write; parser/device-oriented |
| `crml::device::fpga { ... }` | Device block; parser/dispatch |
| `loss crml::backpropagation { ... } params, loss_out =` | Backprop statement; parser |

Script device declarations and flow routing interact with remote discovery and
dispatch. See the [usage guide](USAGE_GUIDE.md) before attempting
device execution, then work through the
remote worker guide for complete programs.

## 13. Style and debugging guidelines

- Use one operation per assignment while learning.
- Choose names that describe values (`product`, `activated`, `total`), not steps.
- Keep quantization directives together at the top.
- Make matrix shapes obvious in comments or input documentation.
- Prefer checked-in, deterministic examples.
- Run `caramel-lint` before execution.
- Read an “unsupported op” error as a backend capability issue, not necessarily
  invalid language syntax.
- When debugging RPN, write the stack after every token.

## 14. Progressive labs

### Lab 1: scalar pipeline

Write a flow computing `(a + b) * c`. Predict the result for `a=2`, `b=3`, `c=4`.

### Lab 2: activation pipeline

Extend `examples/matmul_relu.crml` so the flow also returns
`tensor_sum(result)` as `total`.

### Lab 3: clocked graph

Write a two-clock flow where clock 1 independently computes `a+b` and `c*d`,
and clock 2 adds both intermediate results.

### Lab 4: capability review

Classify `relu`, `conv2d`, a device block, and `@autodiff` using this guide's
feature-status labels. Explain why parse support alone is insufficient.

## 15. Solutions

### Lab 1

```crml
calc::lambda_flow arithmetic(a, b, c) {
    a b add sum =
    sum c mul result =
} return result;
```

The result is `20`.

### Lab 2

```crml
shifted relu result =
result tensor_sum total =
} return result, total;
```

With the guide's sample inputs, `total` is `91`.

### Lab 3

```crml
calc::lambda_flow staged(a, b, c, d) {
    @clock(1):
    a b add left =
    c d mul right =
    @clock(2):
    left right add result =
} return result;
```

### Lab 4

- `relu`: parser and local runtime.
- `conv2d`: parser/operation registry; current local evaluator does not implement it.
- Device block: parser and device/dispatch path.
- `@autodiff`: parser metadata; no general local autodiff execution is implied.

For complete functional numerical programs, see the Markov-chain, Hidden Markov
Model, and Kalman examples in the
remote worker guide.

## Glossary

| Term | Meaning |
|---|---|
| RPN/postfix | Operands precede the operation that consumes them. |
| Flow | Named computation with inputs and returned bindings. |
| Binding | A name assigned to a computed value. |
| Dataflow graph | Dependency graph produced from a parsed flow. |
| Backend | Interpreter, MLIR transformation path, or remote device worker. |
| Quantization | Mapping values into a constrained numeric representation. |
| CRPK/CRRS | Caramel remote request/response binary formats. |
