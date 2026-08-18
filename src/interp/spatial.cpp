// ============================================================================
// Caramel Language - Interpreter spatial ops (pooling, conv)
// ----------------------------------------------------------------------------
// Ticket:  x86_059 (conv/pool/norm)
// Version: 1.0.0
// ============================================================================
#include "caramel/interp/spatial.h"

#include <cstdint>
#include <limits>
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

// Decompose a rank-2 [H,W] or rank-3 [C,H,W] tensor. Returns false otherwise.
bool as_chw(const Value &v, int64_t &C, int64_t &H, int64_t &W) {
  if (v.dims.size() == 2) { C = 1; H = v.dims[0]; W = v.dims[1]; return true; }
  if (v.dims.size() == 3) { C = v.dims[0]; H = v.dims[1]; W = v.dims[2]; return true; }
  return false;
}

std::optional<Value> pool(const Value &x, bool is_max, int64_t k, int64_t s,
                          std::string *error) {
  int64_t C, H, W;
  if (!as_chw(x, C, H, W)) { if (error) *error = "pool needs a rank-2/3 tensor"; return std::nullopt; }
  if (k <= 0 || s <= 0) { if (error) *error = "pool kernel_size/stride must be > 0"; return std::nullopt; }
  if (H < k || W < k) { if (error) *error = "pool window larger than input"; return std::nullopt; }
  const int64_t OH = (H - k) / s + 1, OW = (W - k) / s + 1;
  Value r;
  if (x.dims.size() == 2) r.dims = {OH, OW};
  else r.dims = {C, OH, OW};
  r.data.resize(static_cast<size_t>(C * OH * OW));
  for (int64_t c = 0; c < C; ++c)
    for (int64_t oh = 0; oh < OH; ++oh)
      for (int64_t ow = 0; ow < OW; ++ow) {
        int64_t acc = is_max ? std::numeric_limits<int64_t>::min() : 0;
        for (int64_t kh = 0; kh < k; ++kh)
          for (int64_t kw = 0; kw < k; ++kw) {
            int64_t v = x.data[(c * H + (oh * s + kh)) * W + (ow * s + kw)];
            if (is_max) { if (v > acc) acc = v; } else { acc += v; }
          }
        if (!is_max) acc /= (k * k);  // integer mean, trunc toward zero
        r.data[(c * OH + oh) * OW + ow] = acc;
      }
  return r;
}

// conv2d: x [Cin,H,W] (or [H,W]) * w [Cout,Cin,KH,KW] -> y [Cout,OH,OW], integer
// accumulation (int64, like matmul), zero padding. Mirrors the worker exactly.
std::optional<Value> conv2d(const Value &x, const Value &w, int64_t s, int64_t p,
                            std::string *error) {
  int64_t Cin, H, W;
  if (!as_chw(x, Cin, H, W)) { if (error) *error = "conv2d input must be rank-2/3"; return std::nullopt; }
  if (w.dims.size() != 4) { if (error) *error = "conv2d weight must be [Cout,Cin,KH,KW]"; return std::nullopt; }
  const int64_t Cout = w.dims[0], WCin = w.dims[1], KH = w.dims[2], KW = w.dims[3];
  if (WCin != Cin) { if (error) *error = "conv2d in-channel mismatch"; return std::nullopt; }
  if (s <= 0) { if (error) *error = "conv2d stride must be > 0"; return std::nullopt; }
  const int64_t OH = (H + 2 * p - KH) / s + 1, OW = (W + 2 * p - KW) / s + 1;
  if (OH <= 0 || OW <= 0) { if (error) *error = "conv2d output is empty"; return std::nullopt; }
  Value r;
  r.dims = {Cout, OH, OW};
  r.data.assign(static_cast<size_t>(Cout * OH * OW), 0);
  for (int64_t co = 0; co < Cout; ++co)
    for (int64_t oh = 0; oh < OH; ++oh)
      for (int64_t ow = 0; ow < OW; ++ow) {
        int64_t acc = 0;
        for (int64_t ci = 0; ci < Cin; ++ci)
          for (int64_t kh = 0; kh < KH; ++kh)
            for (int64_t kw = 0; kw < KW; ++kw) {
              const int64_t ih = oh * s - p + kh, iw = ow * s - p + kw;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;  // zero pad
              acc += x.data[(ci * H + ih) * W + iw] *
                     w.data[((co * Cin + ci) * KH + kh) * KW + kw];
            }
        r.data[(co * OH + oh) * OW + ow] = acc;
      }
  return r;
}

}  // namespace

ParamOpEvaluator spatialOpEvaluator() {
  return [](const std::string &op, const std::vector<Value> &in,
            const std::vector<std::pair<std::string, int64_t>> &params,
            std::string *error) -> std::optional<Value> {
    if (op == "maxpool2d" || op == "avgpool2d") {
      if (in.size() != 1) { if (error) *error = op + " expects 1 operand"; return std::nullopt; }
      int64_t k = paramOr(params, "kernel_size", 0);
      int64_t s = paramOr(params, "stride", k);
      return pool(in[0], op == "maxpool2d", k, s, error);
    }
    if (op == "conv2d") {
      if (in.size() != 2) { if (error) *error = "conv2d expects 2 operands (input, weight)"; return std::nullopt; }
      int64_t s = paramOr(params, "stride", 1);
      int64_t p = paramOr(params, "padding", 0);
      return conv2d(in[0], in[1], s, p, error);
    }
    return std::nullopt;
  };
}

}  // namespace caramel::interp
