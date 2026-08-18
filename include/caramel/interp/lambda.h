// ============================================================================
// Caramel Language - CPU lambda-calculus evaluator
// ----------------------------------------------------------------------------
// Ticket:   lang_017 (CPU-Based Lambda Evaluator)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// A small call-by-value lambda-calculus evaluator: abstraction, application,
// environment-captured closures, and curried integer primitives. This is the
// functional core behind Caramel's combinator-based control flow (kestrel/kite
// booleans, conditionals). It also provides the comparison/logic op evaluator
// that plugs into the dataflow Interpreter (lang_016) so parsed programs using
// `>`, `==`, `and`, ... execute.
//
// Integer-only: literals and primitive results are int64_t (Caramel is quantized
// integer). Full Y-combinator recursion / general tensor prims are out of scope
// here (tensor ops are lang_019).
// ============================================================================
#ifndef CARAMEL_INTERP_LAMBDA_H
#define CARAMEL_INTERP_LAMBDA_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "caramel/interp/interpreter.h"  // OpEvaluator

namespace caramel::interp {

// ---- Term: the lambda-calculus AST ----------------------------------------
struct Term;
using TermPtr = std::shared_ptr<const Term>;

enum class TKind { Var, Lit, Abs, App, Prim };

struct Term {
  TKind kind;
  std::string name;     // Var name / Prim name
  int64_t lit = 0;      // Lit value
  std::string param;    // Abs bound variable
  TermPtr body;         // Abs body
  TermPtr fn, arg;      // App function / argument
};

// Smart constructors.
TermPtr var(std::string n);
TermPtr lit(int64_t v);
TermPtr abs(std::string param, TermPtr body);
TermPtr app(TermPtr fn, TermPtr arg);
TermPtr prim(std::string name);          // curried binary primitive (or unary "not")
TermPtr apply(TermPtr fn, std::vector<TermPtr> args);  // left-assoc application

// Common combinators as terms.
TermPtr combinator(const std::string &name);  // identity, kestrel, kite, compose

// ---- Evaluation ------------------------------------------------------------
struct Value64 {  // an evaluated lambda result: an integer or a function
  bool is_int = true;
  int64_t i = 0;
  // function part (closure or partially-applied primitive) is held opaquely by
  // the evaluator; callers normally extract `i` from a fully-applied program.
};

struct EvalResult {
  std::optional<int64_t> value;   // set when the program reduced to an integer
  std::optional<std::string> error;
  bool ok() const { return value.has_value() && !error.has_value(); }
};

// Evaluate a closed term to an integer (call-by-value). Free variables may be
// supplied via `env`.
EvalResult eval(const TermPtr &term,
                const std::unordered_map<std::string, int64_t> &env = {});

// ---- Op evaluator for the dataflow Interpreter (lang_016 seam) -------------
// Handles comparison (>,<,==,>=,<=,!=) and logic (and,or,not,xor) on integer
// scalars/tensors (booleans are 1/0).
OpEvaluator lambdaOpEvaluator();

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_LAMBDA_H
