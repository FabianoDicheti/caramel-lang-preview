// ============================================================================
// Caramel Language - Interpreter normalization ops (layernorm)
// ----------------------------------------------------------------------------
// Ticket:  x86_063 (norms)
// Version: 1.0.0
// ============================================================================
#include "caramel/interp/norm.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "caramel/interp/caramel_norm.h"  // shared bit-exact integer kernel

namespace caramel::interp {

namespace {

int64_t pow10(int q) {
  if (q < 0 || q > 18) return 0;
  int64_t m = 1;
  for (int k = 0; k < q; ++k) m *= 10;
  return m;
}

// layernorm over the last axis (contiguous groups in row-major order), matching
// the worker element-for-element via cnrm_layernorm_group.
std::optional<Value> layernorm(const Value &a, int64_t M, std::string *error) {
  if (a.dims.empty()) { if (error) *error = "layernorm needs a tensor"; return std::nullopt; }
  const int64_t len = a.dims.back();
  const int64_t n = static_cast<int64_t>(a.data.size());
  if (len <= 0 || n % len != 0) {
    if (error) *error = "layernorm: degenerate last axis"; return std::nullopt;
  }
  Value r;
  r.dims = a.dims;
  r.data.resize(a.data.size());
  std::vector<int32_t> in(static_cast<size_t>(len)), out(static_cast<size_t>(len));
  for (int64_t g = 0; g < n; g += len) {
    for (int64_t t = 0; t < len; ++t)
      in[static_cast<size_t>(t)] = static_cast<int32_t>(a.data[g + t]);
    cnrm_layernorm_group(in.data(), out.data(), static_cast<int32_t>(len), M);
    for (int64_t t = 0; t < len; ++t)
      r.data[g + t] = out[static_cast<size_t>(t)];
  }
  return r;
}

}  // namespace

ParamOpEvaluator normOpEvaluator(int quantres) {
  const int64_t M = pow10(quantres);
  return [M](const std::string &op, const std::vector<Value> &in,
             const std::vector<std::pair<std::string, int64_t>> &,
             std::string *error) -> std::optional<Value> {
    if (op != "layernorm") return std::nullopt;
    if (M == 0) { if (error) *error = "layernorm: quantres out of range"; return std::nullopt; }
    if (in.size() != 1) { if (error) *error = "layernorm expects 1 operand"; return std::nullopt; }
    return layernorm(in[0], M, error);
  };
}

}  // namespace caramel::interp
