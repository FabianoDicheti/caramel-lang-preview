// ============================================================================
// Caramel Language - Interpreter activation ops (bit-exact fixed-point)
// ----------------------------------------------------------------------------
// Ticket:  x86_058 (activations) + x86_059 (softmax axis param, x86_057 channel)
// Version: 1.1.0
// ============================================================================
#include "caramel/interp/activation.h"

#include <cstdint>
#include <string>
#include <vector>

#include "caramel/interp/caramel_activation.h"  // shared bit-exact LUT kernel

namespace caramel::interp {

namespace {

// Elementwise unary through the shared LUT (sigmoid/tanh).
std::optional<Value> unary(const cact_table_t &t, const Value &a, int64_t M) {
  Value r;
  r.dims = a.dims;
  r.data.resize(a.data.size());
  for (size_t e = 0; e < a.data.size(); ++e)
    r.data[e] = cact_unary(&t, static_cast<int32_t>(a.data[e]), M);
  return r;
}

// softmax over `axis` (default = last), strided exactly like the worker so the
// two agree element-for-element via cact_softmax_group.
std::optional<Value> softmax(const Value &a, int64_t axisParam, int64_t M,
                             std::string *error) {
  const int64_t rank = static_cast<int64_t>(a.dims.size());
  if (rank == 0) { if (error) *error = "softmax needs a tensor operand"; return std::nullopt; }
  int64_t ax = (axisParam < 0) ? rank - 1 : axisParam;
  if (ax >= rank) { if (error) *error = "softmax axis out of range"; return std::nullopt; }
  const int64_t len = a.dims[ax];
  if (len <= 0) { if (error) *error = "softmax: degenerate axis"; return std::nullopt; }
  int64_t I = 1, O = 1;
  for (int64_t k = ax + 1; k < rank; ++k) I *= a.dims[k];
  for (int64_t k = 0; k < ax; ++k) O *= a.dims[k];
  Value r;
  r.dims = a.dims;
  r.data.resize(a.data.size());
  std::vector<int32_t> in(static_cast<size_t>(len)), out(static_cast<size_t>(len));
  for (int64_t o = 0; o < O; ++o) {
    for (int64_t ii = 0; ii < I; ++ii) {
      const int64_t base = o * len * I + ii;
      for (int64_t t = 0; t < len; ++t)
        in[static_cast<size_t>(t)] = static_cast<int32_t>(a.data[base + t * I]);
      cact_softmax_group(in.data(), out.data(), static_cast<int32_t>(len), M);
      for (int64_t t = 0; t < len; ++t)
        r.data[base + t * I] = out[static_cast<size_t>(t)];
    }
  }
  return r;
}

int64_t paramOr(const std::vector<std::pair<std::string, int64_t>> &ps,
                const std::string &key, int64_t dflt) {
  for (const auto &kv : ps) if (kv.first == key) return kv.second;
  return dflt;
}

}  // namespace

ParamOpEvaluator activationOpEvaluator(int quantres) {
  const int64_t M = cact_pow10(quantres);
  return [M](const std::string &op, const std::vector<Value> &in,
             const std::vector<std::pair<std::string, int64_t>> &params,
             std::string *error) -> std::optional<Value> {
    if (op != "sigmoid" && op != "tanh" && op != "softmax") return std::nullopt;
    if (M == 0) {
      if (error)
        *error = op + ": quantres out of range for fixed-point activation "
                      "(must be 0.." + std::to_string(CACT_MAX_QRES) + ")";
      return std::nullopt;
    }
    if (in.size() != 1) {
      if (error) *error = op + " expects 1 operand";
      return std::nullopt;
    }
    if (op == "sigmoid") return unary(CACT_SIGMOID, in[0], M);
    if (op == "tanh") return unary(CACT_TANH, in[0], M);
    return softmax(in[0], paramOr(params, "axis", -1), M, error);
  };
}

}  // namespace caramel::interp
