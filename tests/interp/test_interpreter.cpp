// ============================================================================
// Caramel Language - Interpreter tests (parse -> dataflow -> execute)
// ----------------------------------------------------------------------------
// Ticket: lang_016 (Interpreter Architecture Design)
// ============================================================================
#include "caramel/interp/interpreter.h"
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

// Constant-folded arithmetic flow: result = (5 + 3) ... but with params.
static void test_scalar_arith() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=0;\n"
      "calc::lambda_flow f(a, b, c) {\n"
      "  a b add s =\n"
      "  s c mul result =\n"
      "} return result;\n",
      keep);
  Interpreter interp;
  interp.set_input("a", Value::scalar(5));
  interp.set_input("b", Value::scalar(3));
  interp.set_input("c", Value::scalar(4));
  auto r = interp.run(g);
  CHECK(r.ok());
  CHECK(r.outputs.count("result") == 1);
  // (5 + 3) * 4 = 32
  CHECK(r.outputs["result"].is_scalar());
  CHECK(r.outputs["result"].data[0] == 32);
}

// Elementwise tensor add + relu.
static void test_tensor_elementwise() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=0;\n"
      "calc::lambda_flow g(x, y) {\n"
      "  x y elemwise_add s =\n"
      "  s relu result =\n"
      "} return result;\n",
      keep);
  Interpreter interp;
  interp.set_input("x", Value::tensor({2, 2}, {1, -5, 3, -2}));
  interp.set_input("y", Value::tensor({2, 2}, {-4, 2, 1, 1}));
  auto r = interp.run(g);
  CHECK(r.ok());
  // add -> {-3, -3, 4, -1}; relu -> {0, 0, 4, 0}
  const auto &out = r.outputs["result"];
  CHECK((out.dims == std::vector<int64_t>{2, 2}));
  CHECK((out.data == std::vector<int64_t>{0, 0, 4, 0}));
}

// A custom evaluator can extend op coverage (mechanism for lang_019 matmul).
static void test_custom_evaluator() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantres=0;\n"
      "calc::lambda_flow h(a, b) {\n"
      "  a b matmul result =\n"
      "} return result;\n",
      keep);
  Interpreter interp;
  interp.set_input("a", Value::scalar(6));
  interp.set_input("b", Value::scalar(7));
  // Stand-in matmul evaluator (scalar product) to prove the plug-in path.
  interp.add_evaluator([](const std::string &op, const std::vector<Value> &in,
                          std::string *) -> std::optional<Value> {
    if (op == "matmul" && in.size() == 2 && in[0].is_scalar() && in[1].is_scalar())
      return Value::scalar(in[0].data[0] * in[1].data[0]);
    return std::nullopt;
  });
  auto r = interp.run(g);
  CHECK(r.ok());
  CHECK(r.outputs["result"].data[0] == 42);
}

static void test_missing_input_errors() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantres=0;\n"
      "calc::lambda_flow f(a, b) {\n  a b add result =\n} return result;\n",
      keep);
  Interpreter interp;
  interp.set_input("a", Value::scalar(1));  // b missing
  auto r = interp.run(g);
  CHECK(!r.ok());
  CHECK(r.error.has_value());
}

int main() {
  test_scalar_arith();
  test_tensor_elementwise();
  test_custom_evaluator();
  test_missing_input_errors();
  if (g_failures == 0) {
    std::printf("OK: all interpreter tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
