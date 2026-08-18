// ============================================================================
// Caramel Language - Interpreter shape ops (concat)
// ----------------------------------------------------------------------------
// Ticket:  x86_059 (shape ops)
// Version: 1.0.0
// ============================================================================
#include "caramel/interp/shape.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace caramel::interp {

namespace {

int64_t paramOr(const std::vector<std::pair<std::string, int64_t>> &ps,
                const std::string &key, int64_t dflt) {
  for (const auto &kv : ps) if (kv.first == key) return kv.second;
  return dflt;
}

// concat a, b along `ax`: same rank, matching dims off the axis; output axis dim
// is the sum. Row-major slab copy, matching the worker element-for-element.
std::optional<Value> concat(const Value &a, const Value &b, int64_t ax,
                            std::string *error) {
  const int64_t R = static_cast<int64_t>(a.dims.size());
  if (R == 0 || a.dims.size() != b.dims.size()) {
    if (error) *error = "concat operands must share rank"; return std::nullopt;
  }
  if (ax < 0 || ax >= R) { if (error) *error = "concat axis out of range"; return std::nullopt; }
  for (int64_t k = 0; k < R; ++k)
    if (k != ax && a.dims[k] != b.dims[k]) {
      if (error) *error = "concat dims must match off the axis"; return std::nullopt;
    }
  int64_t inner = 1, outer = 1;
  for (int64_t k = ax + 1; k < R; ++k) inner *= a.dims[k];
  for (int64_t k = 0; k < ax; ++k) outer *= a.dims[k];
  const int64_t Aax = a.dims[ax], Bax = b.dims[ax], Oax = Aax + Bax;
  Value r;
  r.dims = a.dims;
  r.dims[ax] = Oax;
  r.data.resize(static_cast<size_t>(outer * Oax * inner));
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t t = 0; t < Aax * inner; ++t)
      r.data[o * Oax * inner + t] = a.data[o * Aax * inner + t];
    for (int64_t t = 0; t < Bax * inner; ++t)
      r.data[o * Oax * inner + Aax * inner + t] = b.data[o * Bax * inner + t];
  }
  return r;
}

}  // namespace

ParamOpEvaluator shapeOpEvaluator() {
  return [](const std::string &op, const std::vector<Value> &in,
            const std::vector<std::pair<std::string, int64_t>> &params,
            std::string *error) -> std::optional<Value> {
    if (op == "concat") {
      if (in.size() != 2) { if (error) *error = "concat expects 2 operands"; return std::nullopt; }
      return concat(in[0], in[1], paramOr(params, "axis", 0), error);
    }
    return std::nullopt;
  };
}

}  // namespace caramel::interp
