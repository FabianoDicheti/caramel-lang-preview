// ============================================================================
// Caramel Language - INT8 / quantized precision emulation
// ----------------------------------------------------------------------------
// Ticket:  lang_018 (INT8 Precision Emulation)
// Version: 1.0.0
// ============================================================================
#include "caramel/interp/int8_emulation.h"

#include "caramel/types/type.h"

#include <limits>

namespace caramel::interp {

PrecisionRange precisionRange(double quantmax, double quantmin, int quantres) {
  auto q = caramel::types::QuantParams::from_range(quantmax, quantmin, quantres);
  return PrecisionRange{q.min_int, q.max_int};
}

PrecisionRange intRange(int bits) {
  if (bits <= 0) return PrecisionRange{0, 0};
  if (bits >= 64)
    return PrecisionRange{std::numeric_limits<int64_t>::min(),
                          std::numeric_limits<int64_t>::max()};
  const int64_t hi = (int64_t(1) << (bits - 1)) - 1;
  const int64_t lo = -(int64_t(1) << (bits - 1));
  return PrecisionRange{lo, hi};
}

void saturate(Value &v, PrecisionRange r) {
  for (auto &e : v.data) {
    if (e < r.lo) e = r.lo;
    else if (e > r.hi) e = r.hi;
  }
}

std::function<void(Value &)> saturationTransform(PrecisionRange r) {
  return [r](Value &v) { saturate(v, r); };
}

}  // namespace caramel::interp
