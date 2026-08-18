// ============================================================================
// Caramel Language - INT8 precision emulation tests
// ----------------------------------------------------------------------------
// Ticket: lang_018 (INT8 Precision Emulation)
// ============================================================================
#include "caramel/interp/int8_emulation.h"
#include "caramel/interp/interpreter.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace caramel::parse;
using namespace caramel::interp;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

static caramel::ir::DataflowGraph graph_of(
    const std::string &src, std::unique_ptr<caramel::ast::Program> &keep) {
  Lexer lx(src);
  Parser p(lx.tokenize());
  keep = p.parse();
  for (auto &it : keep->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      return caramel::ir::buildDataflow(
          *static_cast<caramel::ast::LambdaFlow *>(it.get()));
  return {};
}

static void test_ranges() {
  // quantmax=10, quantmin=-10, quantres=0 -> [-10, 10]
  auto r = precisionRange(10, -10, 0);
  CHECK(r.lo == -10 && r.hi == 10);
  // quantres=2 scales by 100 -> [-1000, 1000]
  auto r2 = precisionRange(10, -10, 2);
  CHECK(r2.lo == -1000 && r2.hi == 1000);
  // INT8 signed range
  auto i8 = intRange(8);
  CHECK(i8.lo == -128 && i8.hi == 127);
  auto i64 = intRange(64);
  CHECK(i64.lo == std::numeric_limits<int64_t>::min());
  CHECK(i64.hi == std::numeric_limits<int64_t>::max());
  auto too_wide = intRange(128);
  CHECK(too_wide.lo == std::numeric_limits<int64_t>::min());
  CHECK(too_wide.hi == std::numeric_limits<int64_t>::max());
}

static void test_saturate_value() {
  auto v = Value::tensor({4}, {-50, -5, 25, 200});
  saturate(v, PrecisionRange{-10, 10});
  CHECK((v.data == std::vector<int64_t>{-10, -5, 10, 10}));
}

// With saturation installed, an overflowing product clips to quantmax.
static void test_overflow_saturates() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=0;\n"
      "calc::lambda_flow f(a, b) {\n  a b mul result =\n} return result;\n",
      keep);

  // Without saturation: 5 * 5 = 25.
  {
    Interpreter vm;
    vm.set_input("a", Value::scalar(5));
    vm.set_input("b", Value::scalar(5));
    auto r = vm.run(g);
    CHECK(r.ok() && r.outputs["result"].data[0] == 25);
  }
  // With INT saturation to [-10, 10]: 25 clips to 10.
  {
    Interpreter vm;
    vm.set_op_result_transform(saturationTransform(precisionRange(10, -10, 0)));
    vm.set_input("a", Value::scalar(5));
    vm.set_input("b", Value::scalar(5));
    auto r = vm.run(g);
    CHECK(r.ok());
    CHECK(r.outputs["result"].data[0] == 10);  // saturated
  }
}

// Saturation also applies after matmul (INT accumulation then clamp).
static void test_matmul_saturation() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=0;\n"
      "calc::lambda_flow mm(a, b) {\n  a b matmul result =\n} return result;\n",
      keep);
  Interpreter vm;
  vm.add_evaluator(matrixOpEvaluator());
  vm.set_op_result_transform(saturationTransform(precisionRange(100, -100, 0)));
  vm.set_input("a", Value::tensor({2, 2}, {1, 2, 3, 4}));
  vm.set_input("b", Value::tensor({2, 2}, {5, 6, 7, 8}));
  auto r = vm.run(g);
  CHECK(r.ok());
  // raw matmul {19,22,43,50} all within [-100,100] -> unchanged
  CHECK((r.outputs["result"].data == std::vector<int64_t>{19, 22, 43, 50}));
}

int main() {
  test_ranges();
  test_saturate_value();
  test_overflow_saturates();
  test_matmul_saturation();
  if (g_failures == 0) {
    std::printf("OK: all INT8-emulation tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
