// ============================================================================
// Caramel Language - Interpreter value representation
// ----------------------------------------------------------------------------
// Ticket:   lang_016 (Interpreter Architecture Design)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Caramel is integer-only, so an interpreter value is an integer tensor: a shape
// plus row-major data. INT8/INT16/INT32 program values all hold in int64_t here
// (wide enough for INT32 MAC accumulation without overflow); the dtype/scale that
// turns these back into reals is quantization metadata (lang_003), tracked
// alongside but not needed for the arithmetic itself.
// ============================================================================
#ifndef CARAMEL_INTERP_VALUE_H
#define CARAMEL_INTERP_VALUE_H

#include <cstdint>
#include <vector>

namespace caramel::interp {

struct Value {
  std::vector<int64_t> dims;  // shape; empty = scalar (rank 0)
  std::vector<int64_t> data;  // row-major elements

  Value() = default;

  // Scalar constructor.
  static Value scalar(int64_t v) {
    Value x;
    x.data = {v};
    return x;
  }

  // Dense tensor from shape + data (data.size() must equal product(dims)).
  static Value tensor(std::vector<int64_t> shape, std::vector<int64_t> elems) {
    Value x;
    x.dims = std::move(shape);
    x.data = std::move(elems);
    return x;
  }

  bool is_scalar() const { return dims.empty(); }
  int rank() const { return static_cast<int>(dims.size()); }

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    return dims.empty() ? 1 : n;
  }

  bool same_shape(const Value &o) const { return dims == o.dims; }
};

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_VALUE_H
