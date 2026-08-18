// ============================================================================
// Objectives verification - quantization accuracy
// ----------------------------------------------------------------------------
// Objective: quantization overhead < 5% accuracy loss  (README.md:443)
// Compares an FP64 reference matmul against the quantize -> integer-matmul ->
// dequantize path (the real Caramel encode/compute/decode pipeline), and reports
// the relative Frobenius error across a sweep of resolutions.
// ============================================================================
#include "caramel/quant/quantizer.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace caramel::quant;

// Deterministic pseudo-random doubles in [-range, range].
struct LCG {
  uint64_t s = 0x9e3779b97f4a7c15ULL;
  double next(double range) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = ((s >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);  // [0,1)
    return (u * 2.0 - 1.0) * range;
  }
};

int main() {
  const int M = 8, K = 8, N = 8;
  const double range = 4.0;
  LCG rng;
  std::vector<double> A(M * K), B(K * N);
  for (auto &x : A) x = rng.next(range);
  for (auto &x : B) x = rng.next(range);

  // FP64 reference matmul.
  std::vector<double> ref(M * N, 0.0);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j)
        ref[i * N + j] += A[i * K + k] * B[k * N + j];

  double best_err = 1e18;
  int best_res = 0;
  for (int res : {1, 2, 3, 4}) {
    // A range a bit wider than the data so nothing saturates.
    Quantizer q(range + 1.0, -(range + 1.0), res);
    const double m = double(q.params().multiplier);

    std::vector<int64_t> Ai(M * K), Bi(K * N);
    for (int t = 0; t < M * K; ++t) Ai[t] = q.encode(A[t]);
    for (int t = 0; t < K * N; ++t) Bi[t] = q.encode(B[t]);

    // Integer matmul: result is in units of m^2.
    std::vector<int64_t> Ci(M * N, 0);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
          Ci[i * N + j] += Ai[i * K + k] * Bi[k * N + j];

    // Dequantize (divide by m^2) and measure relative Frobenius error.
    double num = 0.0, den = 0.0;
    for (int t = 0; t < M * N; ++t) {
      double dq = double(Ci[t]) / (m * m);
      num += (dq - ref[t]) * (dq - ref[t]);
      den += ref[t] * ref[t];
    }
    double rel = std::sqrt(num) / std::sqrt(den);
    std::printf("RESULT quant_relerr_res%d MEASURE %.5f\n", res, rel);
    if (rel < best_err) { best_err = rel; best_res = res; }
  }

  bool ok = best_err < 0.05;  // < 5% accuracy loss
  std::printf("RESULT quant_accuracy_under_5pct %s best_relerr=%.5f at_res=%d target<0.05\n",
              ok ? "PASS" : "FAIL", best_err, best_res);
  return ok ? 0 : 1;
}
