// ============================================================================
// Caramel Language - Tensor constructors (generators)
// ----------------------------------------------------------------------------
// The built-in tensor constructors: zeros/ones/full/eye/diag/range/random/
// band/tridiag/from_spectrum. RPN postfix, e.g. `8 8 zeros`, `8 eye`,
// `[2,3,5] diag`.
//
// This is the single implementation, shared by both positions a constructor can
// appear in:
//   * initializer position  `in::m = 8 8 zeros;`   evaluated at load time from
//     literal AST arguments;
//   * flow-body position    `8 8 zeros m =`        evaluated by the interpreter
//     from operand Values.
// Keeping one implementation is what stops the two positions from drifting into
// different semantics for the same syntax.
//
// Everything produced is an ordinary integer `interp::Value`, so a generated
// tensor is indistinguishable downstream from an `in::` literal — nothing about
// the runtime, the wire format, or the worker changes.
// ============================================================================
#ifndef CARAMEL_INTERP_GENERATORS_H
#define CARAMEL_INTERP_GENERATORS_H

#include <optional>
#include <string>
#include <vector>

#include "caramel/interp/value.h"

namespace caramel::interp {

// True if `op` names a tensor constructor.
bool isGenerator(const std::string &op);

// Arity of a constructor, or -1 if `op` is not one. Vector-taking constructors
// (diag, from_spectrum) count their vector as one argument.
int generatorArity(const std::string &op);

// Evaluate a constructor from already-evaluated operands, in RPN order (so
// `8 8 zeros` arrives as {8, 8}). A vector argument arrives as a rank-1 Value;
// scalar arguments as rank-0.
//
// Returns std::nullopt and sets *err (when non-null) if `op` is not a
// constructor, the arity is wrong, an argument has the wrong rank, or a
// dimension is negative. Never wraps or truncates silently.
std::optional<Value> evalGenerator(const std::string &op,
                                   const std::vector<Value> &args,
                                   std::string *err);

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_GENERATORS_H
