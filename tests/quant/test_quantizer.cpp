// ============================================================================
// Caramel Language - Quantization engine tests
// ----------------------------------------------------------------------------
// Ticket: lang_022 (Quantization Engine)
// ============================================================================
#include "caramel/quant/quantizer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace caramel::quant;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// Spec example: 123.4567 at quantres=4 -> 1234567 (just removes the decimal point).
static void test_encode_decode_spec_example() {
  Quantizer q(260.0, -2.0, 4);  // multiplier 10000
  CHECK(q.encode(123.4567) == 1234567);
  CHECK(std::abs(q.decode(1234567) - 123.4567) < 1e-9);
  // enough bits to hold the scaled range (exact width is lang_003's concern).
  CHECK(q.params().bit_width >= 22);
  CHECK(q.params().max_int == 2600000 && q.params().min_int == -20000);
}

// Round-trip within the resolution.
static void test_round_trip() {
  Quantizer q(10.0, -10.0, 2);  // 0.01 resolution
  for (double x : {3.14, -5.7, 0.0, 9.99, -9.99}) {
    int64_t i = q.encode(x);
    double back = q.decode(i);
    CHECK(std::abs(back - x) <= 0.01 + 1e-9);
  }
}

// Saturation: values beyond the range clip.
static void test_saturation() {
  Quantizer q(10.0, -10.0, 0);  // range [-10, 10]
  CHECK(q.encode(15.0) == 10);   // clipped to quantmax
  CHECK(q.encode(-50.0) == -10); // clipped to quantmin
}

// Tensor encode/decode.
static void test_tensor() {
  Quantizer q(10.0, -10.0, 1);  // 0.1 resolution, *10
  auto v = q.encode_tensor({1.2, 3.4, 5.6, 7.8}, {2, 2});
  CHECK((v.dims == std::vector<int64_t>{2, 2}));
  CHECK((v.data == std::vector<int64_t>{12, 34, 56, 78}));
  auto back = q.decode_tensor(v);
  CHECK(std::abs(back[0] - 1.2) < 1e-9);
  CHECK(std::abs(back[3] - 7.8) < 1e-9);
}

static void test_tensor_shape_validation() {
  Quantizer q(10.0, -10.0, 1);

  bool threw = false;
  try {
    (void)q.encode_tensor({1.0, 2.0, 3.0}, {2, 2});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    (void)q.encode_tensor({1.0}, {0});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

// Calibration derives the range from data.
static void test_calibration() {
  std::vector<double> data = {-3.5, 0.0, 2.0, 4.25};
  Quantizer q = Quantizer::calibrate(data, 2);  // range ~[-3.5, 4.25]
  CHECK(q.params().min_int == -350);
  CHECK(q.params().max_int == 425);
  // a symmetric calibration pads to +/- max(|min|,|max|) = 4.25
  Quantizer s = Quantizer::calibrate(data, 2, /*symmetric=*/true);
  CHECK(s.params().max_int == 425);
  CHECK(s.params().min_int == -425);
}

int main() {
  test_encode_decode_spec_example();
  test_round_trip();
  test_saturation();
  test_tensor();
  test_tensor_shape_validation();
  test_calibration();
  if (g_failures == 0) {
    std::printf("OK: all quantizer tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
