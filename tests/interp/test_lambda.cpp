// ============================================================================
// Caramel Language - Lambda evaluator tests
// ----------------------------------------------------------------------------
// Ticket: lang_017 (CPU-Based Lambda Evaluator)
// ============================================================================
#include "caramel/interp/lambda.h"
#include "caramel/interp/interpreter.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>

using namespace caramel::interp;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// (lambda x. x) 5  ==  5
static void test_identity_application() {
  auto term = app(combinator("identity"), lit(5));
  auto r = eval(term);
  CHECK(r.ok());
  CHECK(r.value.value() == 5);
}

// kestrel a b = a (true selects the first); kite a b = b (false selects second).
static void test_combinator_booleans() {
  auto k = apply(combinator("kestrel"), {lit(10), lit(20)});
  auto ki = apply(combinator("kite"), {lit(10), lit(20)});
  CHECK(eval(k).value.value() == 10);
  CHECK(eval(ki).value.value() == 20);
}

// compose f g x = f (g x): with f=+1 (add 1), g=*2 (mul 2): (3*2)+1 = 7
static void test_composition_with_prims() {
  // inc = \n. add n 1 ; dbl = \n. mul n 2
  auto inc = abs("n", apply(prim("add"), {var("n"), lit(1)}));
  auto dbl = abs("n", apply(prim("mul"), {var("n"), lit(2)}));
  auto term = apply(combinator("compose"), {inc, dbl, lit(3)});
  auto r = eval(term);
  CHECK(r.ok());
  CHECK(r.value.value() == 7);
}

// Closures capture the environment: \x. (\y. add x y)  applied to 4 then 5 = 9
static void test_closure_capture() {
  auto adder = abs("x", abs("y", apply(prim("add"), {var("x"), var("y")})));
  auto term = apply(adder, {lit(4), lit(5)});
  CHECK(eval(term).value.value() == 9);
}

// Comparison primitive and free variable from env.
static void test_prim_and_env() {
  auto term = apply(prim("gt"), {var("a"), lit(5)});  // a > 5
  CHECK(eval(term, {{"a", 8}}).value.value() == 1);
  CHECK(eval(term, {{"a", 2}}).value.value() == 0);
}

// The comparison/logic op evaluator plugs into the dataflow Interpreter so a
// parsed program with `>` executes.
static void test_dataflow_comparison() {
  using namespace caramel::parse;
  Lexer lx(
      "crml::quantres=0;\n"
      "calc::lambda_flow f(a, b) {\n  a b > result =\n} return result;\n");
  Parser p(lx.tokenize());
  auto prog = p.parse();
  caramel::ir::DataflowGraph g;
  for (auto &it : prog->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      g = caramel::ir::buildDataflow(*static_cast<caramel::ast::LambdaFlow *>(it.get()));
  Interpreter interp;
  interp.add_evaluator(lambdaOpEvaluator());
  interp.set_input("a", Value::scalar(7));
  interp.set_input("b", Value::scalar(3));
  auto r = interp.run(g);
  CHECK(r.ok());
  CHECK(r.outputs["result"].data[0] == 1);  // 7 > 3
}

static void test_dataflow_comparison_scalar_broadcast() {
  using namespace caramel::parse;
  Lexer lx(
      "crml::quantres=0;\n"
      "calc::lambda_flow f(a, threshold) {\n  a threshold > result =\n} return result;\n");
  Parser p(lx.tokenize());
  auto prog = p.parse();
  caramel::ir::DataflowGraph g;
  for (auto &it : prog->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      g = caramel::ir::buildDataflow(*static_cast<caramel::ast::LambdaFlow *>(it.get()));
  Interpreter interp;
  interp.add_evaluator(lambdaOpEvaluator());
  interp.set_input("a", Value::tensor({2, 2}, {1, 5, 2, 9}));
  interp.set_input("threshold", Value::scalar(3));
  auto r = interp.run(g);
  CHECK(r.ok());
  CHECK((r.outputs["result"].dims == std::vector<int64_t>{2, 2}));
  CHECK((r.outputs["result"].data == std::vector<int64_t>{0, 1, 0, 1}));
}

static void test_unbound_variable_errors() {
  auto r = eval(var("z"));  // free, no env
  CHECK(!r.ok());
  CHECK(r.error.has_value());
}

int main() {
  test_identity_application();
  test_combinator_booleans();
  test_composition_with_prims();
  test_closure_capture();
  test_prim_and_env();
  test_dataflow_comparison();
  test_dataflow_comparison_scalar_broadcast();
  test_unbound_variable_errors();
  if (g_failures == 0) {
    std::printf("OK: all lambda-evaluator tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
