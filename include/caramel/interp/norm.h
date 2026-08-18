// ============================================================================
// Caramel Language - Interpreter normalization ops (layernorm)
// ----------------------------------------------------------------------------
// Ticket:  x86_063 (norms)
// Version: 1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// layernorm over the last axis, computed with the shared bit-exact integer
// kernel (caramel/interp/caramel_norm.h) so the result is byte-identical to the
// bark worker. Captures quantres for the decimal fixed-point scale. Affine
// gamma/beta = 1/0 in v1; batchnorm and general normalized_shape are reserved.
// Register with Interpreter::add_param_evaluator.
// ============================================================================
#ifndef CARAMEL_INTERP_NORM_H
#define CARAMEL_INTERP_NORM_H

#include "caramel/interp/interpreter.h"  // ParamOpEvaluator

namespace caramel::interp {

ParamOpEvaluator normOpEvaluator(int quantres);

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_NORM_H
