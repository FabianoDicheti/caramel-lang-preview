// ============================================================================
// Caramel Language - Quantization engine implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_022 (Quantization Engine)
// Version: 1.0.0
// ============================================================================
#include "caramel/quant/quantizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace caramel::quant {

using caramel::interp::Value;
using caramel::types::QuantParams;

namespace {

size_t checked_numel(const std::vector<int64_t> &dims) {
  if (dims.empty())
    throw std::invalid_argument("encode_tensor requires at least one dimension");

  size_t n = 1;
  for (int64_t dim : dims) {
    if (dim <= 0)
      throw std::invalid_argument("encode_tensor dimensions must be positive");
    const auto d = static_cast<size_t>(dim);
    if (n > std::numeric_limits<size_t>::max() / d)
      throw std::overflow_error("encode_tensor shape is too large");
    n *= d;
  }
  return n;
}

}  // namespace

Quantizer::Quantizer(double quantmax, double quantmin, int quantres)
    : q_(QuantParams::from_range(quantmax, quantmin, quantres)) {}

Quantizer Quantizer::calibrate(const std::vector<double> &data, int quantres,
                               bool symmetric) {
  double lo = 0.0, hi = 0.0;
  if (!data.empty()) {
    lo = *std::min_element(data.begin(), data.end());
    hi = *std::max_element(data.begin(), data.end());
  }
  if (symmetric) {
    const double a = std::max(std::abs(lo), std::abs(hi));
    lo = -a;
    hi = a;
  }
  return Quantizer(QuantParams::from_range(hi, lo, quantres));
}

int64_t Quantizer::encode(double x) const {
  int64_t i = static_cast<int64_t>(std::llround(x * static_cast<double>(q_.multiplier)));
  i = std::max(q_.min_int, std::min(q_.max_int, i));  // saturate to range
  return i;
}

double Quantizer::decode(int64_t i) const {
  return static_cast<double>(i) / static_cast<double>(q_.multiplier);
}

Value Quantizer::encode_tensor(const std::vector<double> &data,
                               std::vector<int64_t> dims) const {
  const size_t expected = checked_numel(dims);
  if (data.size() != expected)
    throw std::invalid_argument("encode_tensor data size does not match shape");

  Value v;
  v.dims = std::move(dims);
  v.data.reserve(data.size());
  for (double x : data) v.data.push_back(encode(x));
  return v;
}

std::vector<double> Quantizer::decode_tensor(const Value &v) const {
  std::vector<double> out;
  out.reserve(v.data.size());
  for (int64_t i : v.data) out.push_back(decode(i));
  return out;
}

}  // namespace caramel::quant
