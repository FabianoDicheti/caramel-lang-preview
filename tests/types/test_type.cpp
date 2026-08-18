// ============================================================================
// Caramel Language - Type system tests
// ----------------------------------------------------------------------------
// Ticket:  lang_003 (Tensor Type System)
// Build:   g++ -std=c++17 -I include tests/types/test_type.cpp src/types/type.cpp -o /tmp/test_type && /tmp/test_type
// ----------------------------------------------------------------------------
// Minimal dependency-free assertion harness (the gtest-based suite arrives with
// the build system in lang_008). Validates bit-width derivation against the
// numbers documented in knowledge/language/LANGUAGE_SPEC_RPN.md, plus shape
// inference and broadcasting.
// ============================================================================
#include "caramel/types/type.h"

#include <cstdio>
#include <cstdlib>

using namespace caramel::types;

static int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);          \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

static void test_bit_width() {
  // Asymmetric endpoint requires 23 signed bits; cardinality alone would say 22.
  auto q1 = QuantParams::from_range(260.0, -2.0, 4);
  CHECK(q1.multiplier == 10000);
  CHECK(q1.max_int == 2600000);
  CHECK(q1.min_int == -20000);
  CHECK(q1.bit_width == 23);
  CHECK(q1.dtype() == DType::Int32);

  // Spec: quantmax=1000, quantmin=-1000, quantres=4 -> 25 bits.
  auto q2 = QuantParams::from_range(1000.0, -1000.0, 4);
  CHECK(q2.bit_width == 25);
  CHECK(q2.dtype() == DType::Int32);

  // Small range fits in Int8: quantmax=10, quantmin=-10, quantres=1 -> 8 bits
  // (span 200, ceil(log2(201)) = 8).
  auto q3 = QuantParams::from_range(10.0, -10.0, 1);
  CHECK(q3.max_int == 100);
  CHECK(q3.min_int == -100);
  CHECK(q3.bit_width == 8);
  CHECK(q3.dtype() == DType::Int8);

  // Integer-only tiny range -> Int4: quantmax=10, quantmin=-10, quantres=0
  // (span 20, ceil(log2(21)) = 5 -> Int8 actually). Check boundary into Int8.
  auto q4 = QuantParams::from_range(10.0, -10.0, 0);
  CHECK(q4.bit_width == 5);
  CHECK(q4.dtype() == DType::Int8);

  // Endpoint safety: [-200, 55] has cardinality 256, but does not fit int8.
  auto q5 = QuantParams::from_range(55.0, -200.0, 0);
  CHECK(q5.bit_width == 9);
  CHECK(q5.dtype() == DType::Int16);
}

static void test_shapes() {
  Shape s({2, 3, 4});
  CHECK(s.rank() == 3);
  CHECK(s.is_static());
  CHECK(s.num_elements().value() == 24);

  Shape dyn({Shape::kDynamic, 4});
  CHECK(!dyn.is_static());
  CHECK(!dyn.num_elements().has_value());
}

static void test_broadcast() {
  // [3,1] and [1,4] -> [3,4]
  auto b = broadcast(Shape({3, 1}), Shape({1, 4}));
  CHECK(b.has_value());
  CHECK((b->dims == std::vector<int64_t>{3, 4}));

  // [2,3] and [4,3] -> incompatible
  CHECK(!broadcast(Shape({2, 3}), Shape({4, 3})).has_value());

  // rank-extension: [5] broadcasts with [3,5] -> [3,5]
  auto b2 = broadcast(Shape({5}), Shape({3, 5}));
  CHECK(b2.has_value());
  CHECK((b2->dims == std::vector<int64_t>{3, 5}));
}

static void test_matmul() {
  // [2,3] x [3,4] -> [2,4]
  auto r = infer_matmul(Shape({2, 3}), Shape({3, 4}));
  CHECK(r.ok());
  CHECK((r.value->dims == std::vector<int64_t>{2, 4}));

  // inner mismatch -> error
  auto bad = infer_matmul(Shape({2, 3}), Shape({5, 4}));
  CHECK(!bad.ok());
  CHECK(bad.error.has_value());

  // batched: [8,2,3] x [8,3,4] -> [8,2,4]
  auto batched = infer_matmul(Shape({8, 2, 3}), Shape({8, 3, 4}));
  CHECK(batched.ok());
  CHECK((batched.value->dims == std::vector<int64_t>{8, 2, 4}));
}

static void test_type_str() {
  TensorType t;
  t.dtype = DType::Int8;
  t.shape = Shape({224, 224, 3});
  CHECK(t.str() == "Tensor<Int8, [224, 224, 3]>");
}

int main() {
  test_bit_width();
  test_shapes();
  test_broadcast();
  test_matmul();
  test_type_str();
  if (g_failures == 0) {
    std::printf("OK: all type-system tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
