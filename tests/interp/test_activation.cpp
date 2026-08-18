// ============================================================================
// Caramel Language - Interpreter activation-ops tests
// ----------------------------------------------------------------------------
// Ticket: x86_058 (activations: sigmoid/tanh/softmax)
// ----------------------------------------------------------------------------
// Asserts the interpreter's activationOpEvaluator produces the SAME integer
// outputs the bark worker's ir_exec asserts for identical inputs at q=3 (both
// derive from the byte-identical shared kernel caramel_activation.h), which is
// what makes --verify byte-match. Keep these expected constants in lock-step
// with bark/tests/ir_exec_test.c.
// ============================================================================
#include "caramel/interp/activation.h"
#include "caramel/interp/value.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

using namespace caramel::interp;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

using Params = std::vector<std::pair<std::string, int64_t>>;

static Value apply(const ParamOpEvaluator &ev, const std::string &op, Value in,
                   Params params = {}) {
  std::string err;
  auto r = ev(op, {std::move(in)}, params, &err);
  if (!r) { std::printf("  (%s error: %s)\n", op.c_str(), err.c_str()); return {}; }
  return *r;
}

int main() {
  ParamOpEvaluator ev = activationOpEvaluator(3);  // q=3, M=1000

  // sigmoid: reals 0,1,2,-2 -> 500,731,881,119
  {
    Value r = apply(ev, "sigmoid",
                    Value::tensor({4}, {0, 1000, 2000, -2000}));
    CHECK((r.data == std::vector<int64_t>{500, 731, 881, 119}));
  }
  // tanh: reals 0,1,-1,0.5 -> 0,761,-761,462
  {
    Value r = apply(ev, "tanh", Value::tensor({4}, {0, 1000, -1000, 500}));
    CHECK((r.data == std::vector<int64_t>{0, 761, -761, 462}));
  }
  // softmax (last axis): reals [1,2,3] -> [90,245,665]
  {
    Value r = apply(ev, "softmax", Value::tensor({3}, {1000, 2000, 3000}));
    CHECK((r.data == std::vector<int64_t>{90, 245, 665}));
  }
  // softmax last axis on a 2x2: each ROW normalized independently.
  // row [1,2]->[0.269,0.731]->[269,731]; row [3,4]-> same.
  {
    Value r = apply(ev, "softmax",
                    Value::tensor({2, 2}, {1000, 2000, 3000, 4000}));
    CHECK((r.data == std::vector<int64_t>{269, 731, 269, 731}));
  }
  // softmax {axis:0} on the same 2x2: COLUMNS normalized. col [1,3]->[119,881],
  // col [2,4]->[119,881]. Must match the worker's 2x2 axis-0 vector.
  {
    Value r = apply(ev, "softmax",
                    Value::tensor({2, 2}, {1000, 2000, 3000, 4000}),
                    {{"axis", 0}});
    CHECK((r.data == std::vector<int64_t>{119, 119, 881, 881}));
  }
  // quantres out of range -> evaluator declines with an error (nullopt).
  {
    ParamOpEvaluator bad = activationOpEvaluator(99);
    std::string err;
    auto r = bad("sigmoid", {Value::tensor({1}, {0})}, {}, &err);
    CHECK(!r.has_value() && !err.empty());
  }
  // ops this evaluator does not own pass through (nullopt, no error set).
  {
    std::string err;
    auto r = ev("matmul", {Value::tensor({1}, {0})}, {}, &err);
    CHECK(!r.has_value());
  }

  if (g_failures == 0) {
    std::printf("OK: all activation tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
