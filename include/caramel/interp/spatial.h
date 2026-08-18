// ============================================================================
// Caramel Language - Interpreter spatial ops (pooling, conv)
// ----------------------------------------------------------------------------
// Ticket:  x86_059 (conv/pool/norm)
// Version: 1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Integer maxpool2d/avgpool2d (and conv2d) on quantized tensors, param-aware so
// kernel_size/stride/padding come from the node params (folded into the IR
// immediate for the worker). Results mirror the bark worker element-for-element:
// avgpool integer-divides toward zero; conv accumulates in int64 like matmul.
// Tensor layout: rank-2 [H,W] or rank-3 [C,H,W] (pool/conv over H,W per channel;
// conv weight is [C_out,C_in,KH,KW]). Register with add_param_evaluator.
// ============================================================================
#ifndef CARAMEL_INTERP_SPATIAL_H
#define CARAMEL_INTERP_SPATIAL_H

#include "caramel/interp/interpreter.h"  // ParamOpEvaluator

namespace caramel::interp {

ParamOpEvaluator spatialOpEvaluator();

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_SPATIAL_H
