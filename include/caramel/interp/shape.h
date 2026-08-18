// ============================================================================
// Caramel Language - Interpreter shape ops (concat; reshape/split reserved)
// ----------------------------------------------------------------------------
// Ticket:  x86_059 (shape ops)
// Version: 1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// concat along an axis (from node params, default 0), mirroring the worker.
// reshape (needs a shape param tensor) and split (needs multi-output IR) are
// reserved and tracked separately. Register with add_param_evaluator.
// ============================================================================
#ifndef CARAMEL_INTERP_SHAPE_H
#define CARAMEL_INTERP_SHAPE_H

#include "caramel/interp/interpreter.h"  // ParamOpEvaluator

namespace caramel::interp {

ParamOpEvaluator shapeOpEvaluator();

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_SHAPE_H
