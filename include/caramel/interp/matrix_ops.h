// ============================================================================
// Caramel Language - Interpreter matrix/tensor operations
// ----------------------------------------------------------------------------
// Ticket:   lang_019 (Matrix Operations in Interpreter)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// CPU integer implementations of the tensor ops, exposed as an OpEvaluator that
// plugs into the lang_016 interpreter. Caramel is integer-only, so these compute
// directly on int64 tensor data (INT8/16/32 values, INT32+ accumulation) — a
// faithful functional model of the hardware. Float BLAS/Eigen is intentionally
// not used: it would not match the integer/quantized semantics. SIMD/Eigen-style
// optimization is a follow-on.
//
// Covered: matmul (2D), transpose (2D), scalar_mul, and reductions
// (tensor_sum/mean/max/min). conv2d and the remaining primitives extend the same
// evaluator.
// ============================================================================
#ifndef CARAMEL_INTERP_MATRIX_OPS_H
#define CARAMEL_INTERP_MATRIX_OPS_H

#include "caramel/interp/interpreter.h"  // OpEvaluator

namespace caramel::interp {

// OpEvaluator handling matmul, transpose, scalar_mul, and reductions on integer
// tensors. Register with Interpreter::add_evaluator.
OpEvaluator matrixOpEvaluator();

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_MATRIX_OPS_H
