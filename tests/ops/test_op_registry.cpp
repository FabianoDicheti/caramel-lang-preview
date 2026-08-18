// ============================================================================
// Caramel Language - Primitive operation registry tests
// ----------------------------------------------------------------------------
// Ticket: lang_004 (Primitive Operations)
// Build:  g++ -std=c++17 -I include tests/ops/test_op_registry.cpp
//             src/ops/op_registry.cpp src/types/type.cpp -o /tmp/test_ops
// ============================================================================
#include "caramel/ops/op_signature.h"

#include <cstdio>
#include <cstdlib>

using namespace caramel::ops;
using caramel::types::Shape;

static int g_failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

int main() {
  const OpRegistry& reg = OpRegistry::instance();
  CHECK(reg.size() > 50);

  // Arity lookups drive RPN reduction [A-9].
  CHECK(reg.arity_of("matmul") == 2);
  CHECK(reg.arity_of("relu") == 1);
  CHECK(reg.arity_of("batchnorm") == 5);
  CHECK(reg.arity_of("multihead_attention") == 3);
  CHECK(reg.arity_of("tensor.init") == 0);
  CHECK(reg.arity_of("not_a_real_op") == -1);

  // Tensor constructors (host-side generators): fixed RPN arities.
  CHECK(reg.arity_of("zeros") == 2);
  CHECK(reg.arity_of("ones") == 2);
  CHECK(reg.arity_of("full") == 3);
  CHECK(reg.arity_of("eye") == 1);
  CHECK(reg.arity_of("diag") == 1);
  CHECK(reg.arity_of("range") == 1);
  CHECK(reg.arity_of("random") == 2);
  CHECK(reg.arity_of("band") == 4);
  CHECK(reg.arity_of("tridiag") == 4);
  CHECK(reg.arity_of("from_spectrum") == 1);
  // zeros yields a rank-2 (dynamic) shape; range yields rank-1.
  const Shape scalar0{std::vector<int64_t>{}};
  CHECK(reg.lookup("zeros")->infer_shape({scalar0, scalar0}).value->rank() == 2);
  CHECK(reg.lookup("range")->infer_shape({scalar0}).value->rank() == 1);

  // Routing defaults from the language spec routing table.
  CHECK(reg.lookup("matmul")->target == HwTarget::Fpga);
  CHECK(reg.lookup("softmax")->target == HwTarget::Cpu);
  CHECK(reg.lookup("layernorm")->target == HwTarget::Cpu);

  // Object-parameter positional order [A-7].
  CHECK((reg.lookup("conv2d")->param_keys == std::vector<std::string>{"stride", "padding"}));

  // Shape inference: matmul [2,3] x [3,4] -> [2,4]
  auto mm = reg.lookup("matmul")->infer_shape({Shape({2, 3}), Shape({3, 4})});
  CHECK(mm.ok());
  CHECK((mm.value->dims == std::vector<int64_t>{2, 4}));

  // Shape inference: relu preserves shape
  auto r = reg.lookup("relu")->infer_shape({Shape({8, 8})});
  CHECK(r.ok());
  CHECK((r.value->dims == std::vector<int64_t>{8, 8}));

  // Shape inference: tensor_sum -> scalar
  auto s = reg.lookup("tensor_sum")->infer_shape({Shape({4, 4})});
  CHECK(s.ok());
  CHECK(s.value->rank() == 0);

  // transpose swaps last two dims
  auto t = reg.lookup("transpose")->infer_shape({Shape({2, 3, 4})});
  CHECK(t.ok());
  CHECK((t.value->dims == std::vector<int64_t>{2, 4, 3}));

  if (g_failures == 0) {
    std::printf("OK: all op-registry tests passed (%zu ops registered)\n", reg.size());
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
