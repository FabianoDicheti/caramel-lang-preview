// ============================================================================
// Caramel Language - Interpreter activation ops (bit-exact fixed-point)
// ----------------------------------------------------------------------------
// Ticket:   x86_058 (activations: sigmoid/tanh/softmax)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// sigmoid/tanh/softmax over the quantized integer domain, computed with the
// shared integer LUT kernel (caramel/interp/caramel_activation.h) so the local
// interpreter result is BYTE-IDENTICAL to the bark worker's — a --verify
// requirement. The evaluator captures `quantres` because the decimal
// fixed-point scale (10^quantres) is needed to interpret register integers as
// reals. softmax normalizes over the LAST axis (the interpreter op interface
// carries no axis param; the worker reads a non-default axis from the IR
// immediate). Semantics are normative: PROTOCOL_SPEC.md 6.5 "Activations".
// ============================================================================
#ifndef CARAMEL_INTERP_ACTIVATION_H
#define CARAMEL_INTERP_ACTIVATION_H

#include "caramel/interp/interpreter.h"  // OpEvaluator

namespace caramel::interp {

// Param-aware evaluator for sigmoid/tanh/softmax on integer tensors at the given
// decimal fixed-point resolution. softmax reads {axis:N} from the node params
// (default = last axis), matching the worker's IR-immediate axis. Register with
// Interpreter::add_param_evaluator.
ParamOpEvaluator activationOpEvaluator(int quantres);

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_ACTIVATION_H
