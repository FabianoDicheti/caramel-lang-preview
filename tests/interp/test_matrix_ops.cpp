// ============================================================================
// Caramel Language - Interpreter matrix-ops tests
// ----------------------------------------------------------------------------
// Ticket: lang_019 (Matrix Operations in Interpreter)
// ============================================================================
#include "caramel/interp/interpreter.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>

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

// [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]  (the spec's canonical example)
static void test_matmul_end_to_end() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantres=0;\n"
      "calc::lambda_flow mm(a, b) {\n  a b matmul result =\n} return result;\n",
      keep);
  Interpreter interp;
  interp.add_evaluator(matrixOpEvaluator());
  interp.set_input("a", Value::tensor({2, 2}, {1, 2, 3, 4}));
  interp.set_input("b", Value::tensor({2, 2}, {5, 6, 7, 8}));
  auto r = interp.run(g);
  CHECK(r.ok());
  const auto &out = r.outputs["result"];
  CHECK((out.dims == std::vector<int64_t>{2, 2}));
  CHECK((out.data == std::vector<int64_t>{19, 22, 43, 50}));
}

// Non-square: [2x3] x [3x2] -> [2x2]
static void test_matmul_rectangular() {
  Interpreter interp;
  interp.add_evaluator(matrixOpEvaluator());
  std::string err;
  auto ev = matrixOpEvaluator();
  auto a = Value::tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = Value::tensor({3, 2}, {7, 8, 9, 10, 11, 12});
  auto r = ev("matmul", {a, b}, &err);
  CHECK(r.has_value());
  // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
  // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
  CHECK((r->data == std::vector<int64_t>{58, 64, 139, 154}));
}

static void test_transpose_and_reductions() {
  auto ev = matrixOpEvaluator();
  std::string err;
  auto a = Value::tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto t = ev("transpose", {a}, &err);
  CHECK(t.has_value());
  CHECK((t->dims == std::vector<int64_t>{3, 2}));
  CHECK((t->data == std::vector<int64_t>{1, 4, 2, 5, 3, 6}));

  CHECK(ev("tensor_sum", {a}, &err)->data[0] == 21);
  CHECK(ev("tensor_mean", {a}, &err)->data[0] == 3);  // 21/6
  CHECK(ev("tensor_max", {a}, &err)->data[0] == 6);
  CHECK(ev("tensor_min", {a}, &err)->data[0] == 1);
}

// Matrix ops compose with the lang_016 built-ins (matmul then elemwise add).
static void test_matmul_then_add() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantres=0;\n"
      "calc::lambda_flow f(a, b, bias) {\n"
      "  a b matmul p =\n"
      "  p bias elemwise_add result =\n"
      "} return result;\n",
      keep);
  Interpreter interp;
  interp.add_evaluator(matrixOpEvaluator());  // matmul
  interp.set_input("a", Value::tensor({2, 2}, {1, 2, 3, 4}));
  interp.set_input("b", Value::tensor({2, 2}, {5, 6, 7, 8}));
  interp.set_input("bias", Value::tensor({2, 2}, {1, 1, 1, 1}));
  auto r = interp.run(g);
  CHECK(r.ok());
  // matmul = {19,22,43,50}; + bias -> {20,23,44,51}
  CHECK((r.outputs["result"].data == std::vector<int64_t>{20, 23, 44, 51}));
}

static void test_inner_dim_mismatch_errors() {
  auto ev = matrixOpEvaluator();
  std::string err;
  auto a = Value::tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = Value::tensor({2, 2}, {1, 2, 3, 4});  // inner 3 != 2
  CHECK(!ev("matmul", {a, b}, &err).has_value());
  CHECK(!err.empty());
}

static void test_invalid_operands_error_instead_of_ub() {
  auto ev = matrixOpEvaluator();
  std::string err;

  CHECK(!ev("tensor_sum", {}, &err).has_value());
  CHECK(!err.empty());

  err.clear();
  auto empty = Value::tensor({0}, {});
  CHECK(!ev("tensor_max", {empty}, &err).has_value());
  CHECK(!err.empty());

  err.clear();
  auto malformed = Value::tensor({2, 2}, {1, 2, 3});
  auto valid = Value::tensor({2, 2}, {1, 2, 3, 4});
  CHECK(!ev("matmul", {malformed, valid}, &err).has_value());
  CHECK(!err.empty());

  err.clear();
  CHECK(!ev("transpose", {malformed}, &err).has_value());
  CHECK(!err.empty());
}

int main() {
  test_matmul_end_to_end();
  test_matmul_rectangular();
  test_transpose_and_reductions();
  test_matmul_then_add();
  test_inner_dim_mismatch_errors();
  test_invalid_operands_error_instead_of_ub();
  if (g_failures == 0) {
    std::printf("OK: all matrix-ops tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
