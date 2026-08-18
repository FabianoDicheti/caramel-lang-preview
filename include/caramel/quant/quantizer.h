// ============================================================================
// Caramel Language - Quantization engine (FP32 <-> INT)
// ----------------------------------------------------------------------------
// Ticket:   lang_022 (Quantization Engine)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Host-side post-training quantization: the FP32<->INT encode/decode that crosses
// the boundary into the integer-only program (the Raspberry-Pi-side encode/decode
// in the language spec). Uses the lang_003 "decimal point removal" mapping:
//   encode(x) = clamp(round(x * 10^quantres), min_int, max_int)
//   decode(i) = i / 10^quantres
// Calibration computes the [quantmin, quantmax] range from sample data.
// ============================================================================
#ifndef CARAMEL_QUANT_QUANTIZER_H
#define CARAMEL_QUANT_QUANTIZER_H

#include <cstdint>
#include <vector>

#include "caramel/interp/value.h"
#include "caramel/types/type.h"

namespace caramel::quant {

class Quantizer {
 public:
  // Build from an explicit range (quantmax/quantmin) and resolution.
  Quantizer(double quantmax, double quantmin, int quantres);
  explicit Quantizer(const caramel::types::QuantParams &q) : q_(q) {}

  // Calibrate the range from sample data (min/max), at the given resolution.
  // A symmetric option pads the range to be symmetric about zero.
  static Quantizer calibrate(const std::vector<double> &data, int quantres,
                             bool symmetric = false);

  const caramel::types::QuantParams &params() const { return q_; }
  int bit_width() const { return q_.bit_width; }
  caramel::types::DType dtype() const { return q_.dtype(); }

  // Scalar FP32 <-> INT.
  int64_t encode(double x) const;
  double decode(int64_t i) const;

  // Tensor FP32 -> integer Value, and back.
  caramel::interp::Value encode_tensor(const std::vector<double> &data,
                                       std::vector<int64_t> dims) const;
  std::vector<double> decode_tensor(const caramel::interp::Value &v) const;

 private:
  caramel::types::QuantParams q_;
};

}  // namespace caramel::quant

#endif  // CARAMEL_QUANT_QUANTIZER_H
