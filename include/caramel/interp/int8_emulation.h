// ============================================================================
// Caramel Language - INT8 / quantized precision emulation
// ----------------------------------------------------------------------------
// Ticket:   lang_018 (INT8 Precision Emulation)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// The interpreter computes in int64 (no overflow). To match the hardware's
// quantized datapath exactly, op results are saturated to the program's
// quantization range (the language spec's default "saturating arithmetic: values
// clip to quantmax/quantmin"). The range is derived from the lang_003 quant
// directives. Install via Interpreter::set_op_result_transform.
// ============================================================================
#ifndef CARAMEL_INTERP_INT8_EMULATION_H
#define CARAMEL_INTERP_INT8_EMULATION_H

#include <cstdint>
#include <functional>

#include "caramel/interp/value.h"

namespace caramel::interp {

// The saturating integer range [lo, hi] a value may occupy at runtime.
struct PrecisionRange {
  int64_t lo = 0;
  int64_t hi = 0;
};

// Derive the saturating range from a quantization config (decimal-point-removal,
// lang_003): lo = round(quantmin * 10^quantres), hi = round(quantmax * 10^quantres).
PrecisionRange precisionRange(double quantmax, double quantmin, int quantres);

// The signed range of a fixed INT bit width, e.g. bits=8 -> [-128, 127].
PrecisionRange intRange(int bits);

// Saturating clamp every element of `v` into [r.lo, r.hi] (in place).
void saturate(Value &v, PrecisionRange r);

// A result transform for Interpreter::set_op_result_transform that saturates
// every op output to `r` (emulating the quantized datapath).
std::function<void(Value &)> saturationTransform(PrecisionRange r);

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_INT8_EMULATION_H
