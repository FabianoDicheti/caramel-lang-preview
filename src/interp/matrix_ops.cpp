// ============================================================================
// Caramel Language - Interpreter matrix/tensor operations
// ----------------------------------------------------------------------------
// Ticket:  lang_019 (Matrix Operations in Interpreter)
// Version: 1.0.0
// ============================================================================
#include "caramel/interp/matrix_ops.h"

#include <algorithm>
#include <limits>

namespace caramel::interp {

namespace {

bool is2d(const Value &v) { return v.rank() == 2; }

bool dense2d_ok(const Value &v, std::string *err, const char *op) {
  if (!is2d(v)) {
    if (err) *err = std::string(op) + " requires 2-D operands";
    return false;
  }
  if (v.dims[0] < 0 || v.dims[1] < 0) {
    if (err) *err = std::string(op) + " requires non-negative dimensions";
    return false;
  }
  const auto rows = static_cast<size_t>(v.dims[0]);
  const auto cols = static_cast<size_t>(v.dims[1]);
  if (rows != 0 && cols > std::numeric_limits<size_t>::max() / rows) {
    if (err) *err = std::string(op) + " tensor shape is too large";
    return false;
  }
  if (v.data.size() != rows * cols) {
    if (err) *err = std::string(op) + " tensor data does not match shape";
    return false;
  }
  return true;
}

// [M,K] x [K,N] -> [M,N] integer matmul (INT accumulation).
std::optional<Value> matmul(const Value &a, const Value &b, std::string *err) {
  if (!dense2d_ok(a, err, "matmul") || !dense2d_ok(b, err, "matmul"))
    return std::nullopt;
  const int64_t M = a.dims[0], K = a.dims[1];
  const int64_t K2 = b.dims[0], N = b.dims[1];
  if (K != K2) { if (err) *err = "matmul inner dimensions disagree"; return std::nullopt; }
  Value r;
  r.dims = {M, N};
  r.data.assign(static_cast<size_t>(M * N), 0);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) {
      const int64_t aik = a.data[i * K + k];
      if (aik == 0) continue;
      for (int64_t j = 0; j < N; ++j)
        r.data[i * N + j] += aik * b.data[k * N + j];
    }
  return r;
}

std::optional<Value> transpose(const Value &a, std::string *err) {
  if (!dense2d_ok(a, err, "transpose")) return std::nullopt;
  const int64_t M = a.dims[0], N = a.dims[1];
  Value r;
  r.dims = {N, M};
  r.data.resize(a.data.size());
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      r.data[j * M + i] = a.data[i * N + j];
  return r;
}

std::optional<Value> reduce_scalar(const Value &a, char which, std::string *err) {
  if (a.data.empty() && which != 's') {
    if (err) *err = "cannot reduce an empty tensor";
    return std::nullopt;
  }
  int64_t acc = 0;
  if (which == 'x') acc = std::numeric_limits<int64_t>::min();
  if (which == 'm') acc = std::numeric_limits<int64_t>::max();  // min
  for (int64_t e : a.data) {
    switch (which) {
      case 's': case 'a': acc += e; break;        // sum / mean accumulator
      case 'x': acc = std::max(acc, e); break;     // max
      case 'm': acc = std::min(acc, e); break;     // min
    }
  }
  if (which == 'a') {  // mean (integer division)
    acc /= static_cast<int64_t>(a.data.size());
  }
  // Rank-1 [1], not a rank-0 scalar. CRPK tensors always carry rank >= 1, so
  // the worker commits whole-tensor reductions to dims {1}
  // (bark/kernel/ir_exec.c) and a reduction result comes back off the wire that
  // way. Returning rank-0 here made the two representations diverge, and each
  // divergence needed its own workaround: scalar_mul rejected the wire form,
  // concat rejected the rank-0 form, and verifyAgainstLocal reported a MISMATCH
  // on identical data. Matching the wire retires all three at the source.
  return Value::tensor({1}, {acc});
}

}  // namespace

OpEvaluator matrixOpEvaluator() {
  return [](const std::string &op, const std::vector<Value> &in,
            std::string *error) -> std::optional<Value> {
    if (op == "matmul") {
      if (in.size() != 2) { if (error) *error = "matmul expects 2 operands"; return std::nullopt; }
      return matmul(in[0], in[1], error);
    }
    if (op == "transpose") {
      if (in.size() != 1) { if (error) *error = "transpose expects 1 operand"; return std::nullopt; }
      return transpose(in[0], error);
    }
    if (op == "scalar_mul") {
      if (in.size() != 2) { if (error) *error = "scalar_mul expects 2 operands"; return std::nullopt; }
      // scalar * tensor (either operand order).
      //
      // A rank-0 scalar cannot cross the wire: CRPK tensors always carry rank
      // >= 1, so a scalar reaches the worker as rank-1 [1] and kernel/ir_exec.c
      // accepts it on that basis ("one operand is a rank-1 [1] scalar, either
      // order"). Requiring is_scalar() here therefore rejected locally what the
      // worker computes happily, so any .crml using scalar_mul failed --verify
      // with "scalar_mul needs a scalar operand". Treat a single-element
      // operand as the scalar, matching the worker.
      auto is_scalarish = [](const Value &v) {
        return v.is_scalar() || v.data.size() == 1;
      };
      const Value &s = is_scalarish(in[0]) ? in[0] : in[1];
      const Value &t = is_scalarish(in[0]) ? in[1] : in[0];
      if (!is_scalarish(s)) { if (error) *error = "scalar_mul needs a scalar operand"; return std::nullopt; }
      Value r = t;
      for (auto &e : r.data) e *= s.data[0];
      return r;
    }
    auto reduce = [&](char which) -> std::optional<Value> {
      if (in.size() != 1) { if (error) *error = op + " expects 1 operand"; return std::nullopt; }
      return reduce_scalar(in[0], which, error);
    };
    if (op == "tensor_sum")  return reduce('s');
    if (op == "tensor_mean") return reduce('a');
    if (op == "tensor_max")  return reduce('x');
    if (op == "tensor_min")  return reduce('m');
    return std::nullopt;  // not a matrix op handled here
  };
}

}  // namespace caramel::interp
