// ============================================================================
// Caramel Language - Interpreter golden test suite
// ----------------------------------------------------------------------------
// Ticket: lang_021 (Interpreter Test Suite)
// A comprehensive, data-driven suite running .crml programs end to end (parse ->
// dataflow -> interpret) and comparing outputs against golden reference values,
// across the interpreter op set (scalar/elementwise arithmetic, relu, comparison,
// logic, matmul, transpose, reductions, a full linear layer, and clock sections).
// ============================================================================
#include "caramel/interp/interpreter.h"
#include "caramel/interp/lambda.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace caramel::parse;
using namespace caramel::interp;

static int g_failures = 0;
static int g_cases = 0;

struct NamedValue { std::string name; Value value; };
struct Case {
  std::string name;
  std::string src;
  std::vector<NamedValue> inputs;
  std::vector<NamedValue> expected;  // expected output values by return name
};

static const char *PREAMBLE =
    "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n";

// Run one case; returns true on pass.
static bool run_case(const Case &c) {
  Lexer lx(PREAMBLE + c.src);
  Parser p(lx.tokenize());
  auto prog = p.parse();
  if (!p.ok()) { std::printf("  [%s] parse error\n", c.name.c_str()); return false; }

  const caramel::ast::LambdaFlow *flow = nullptr;
  for (auto &it : prog->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      flow = static_cast<caramel::ast::LambdaFlow *>(it.get());
  if (!flow) { std::printf("  [%s] no flow\n", c.name.c_str()); return false; }

  auto g = caramel::ir::buildDataflow(*flow);
  Interpreter vm;
  vm.add_evaluator(matrixOpEvaluator());
  vm.add_evaluator(lambdaOpEvaluator());
  for (const auto &in : c.inputs) vm.set_input(in.name, in.value);

  auto r = vm.run(g);
  if (!r.ok()) {
    std::printf("  [%s] runtime error: %s\n", c.name.c_str(),
                r.error->message.c_str());
    return false;
  }
  for (const auto &exp : c.expected) {
    auto it = r.outputs.find(exp.name);
    if (it == r.outputs.end()) { std::printf("  [%s] missing output %s\n", c.name.c_str(), exp.name.c_str()); return false; }
    if (it->second.dims != exp.value.dims || it->second.data != exp.value.data) {
      std::printf("  [%s] output %s mismatch\n", c.name.c_str(), exp.name.c_str());
      return false;
    }
  }
  return true;
}

static Value S(int64_t v) { return Value::scalar(v); }
static Value T(std::vector<int64_t> d, std::vector<int64_t> x) {
  return Value::tensor(std::move(d), std::move(x));
}

int main() {
  std::vector<Case> cases = {
    {"scalar_add", "calc::lambda_flow f(a, b) {\n a b add r =\n} return r;\n",
     {{"a", S(5)}, {"b", S(3)}}, {{"r", S(8)}}},

    {"scalar_chain", "calc::lambda_flow f(a, b, c) {\n a b add s =\n s c mul r =\n} return r;\n",
     {{"a", S(5)}, {"b", S(3)}, {"c", S(4)}}, {{"r", S(32)}}},

    {"elemwise_add", "calc::lambda_flow f(x, y) {\n x y elemwise_add r =\n} return r;\n",
     {{"x", T({2,2},{1,2,3,4})}, {"y", T({2,2},{10,20,30,40})}}, {{"r", T({2,2},{11,22,33,44})}}},

    {"elemwise_sub_mul", "calc::lambda_flow f(x, y) {\n x y elemwise_sub d =\n d y elemwise_mul r =\n} return r;\n",
     {{"x", T({2,2},{10,10,10,10})}, {"y", T({2,2},{1,2,3,4})}}, {{"r", T({2,2},{9,16,21,24})}}},

    {"scalar_mul", "calc::lambda_flow f(x, s) {\n x s scalar_mul r =\n} return r;\n",
     {{"x", T({2,2},{1,2,3,4})}, {"s", S(3)}}, {{"r", T({2,2},{3,6,9,12})}}},

    {"relu", "calc::lambda_flow f(x) {\n x relu r =\n} return r;\n",
     {{"x", T({4},{-3,0,5,-1})}}, {{"r", T({4},{0,0,5,0})}}},

    {"comparison_gt", "calc::lambda_flow f(a, b) {\n a b > r =\n} return r;\n",
     {{"a", S(7)}, {"b", S(3)}}, {{"r", S(1)}}},

    {"logic_and", "calc::lambda_flow f(a, b) {\n a b and r =\n} return r;\n",
     {{"a", S(1)}, {"b", S(0)}}, {{"r", S(0)}}},

    {"matmul_square", "calc::lambda_flow f(a, b) {\n a b matmul r =\n} return r;\n",
     {{"a", T({2,2},{1,2,3,4})}, {"b", T({2,2},{5,6,7,8})}}, {{"r", T({2,2},{19,22,43,50})}}},

    {"matmul_rect", "calc::lambda_flow f(a, b) {\n a b matmul r =\n} return r;\n",
     {{"a", T({2,3},{1,2,3,4,5,6})}, {"b", T({3,2},{7,8,9,10,11,12})}}, {{"r", T({2,2},{58,64,139,154})}}},

    {"linear_layer", "calc::lambda_flow f(x, w, bias) {\n x w matmul p =\n p bias elemwise_add s =\n s relu r =\n} return r;\n",
     {{"x", T({2,2},{1,2,3,4})}, {"w", T({2,2},{5,6,7,8})}, {"bias", T({2,2},{0,0,-100,0})}},
     {{"r", T({2,2},{19,22,0,50})}}},

    {"transpose", "calc::lambda_flow f(x) {\n x transpose r =\n} return r;\n",
     {{"x", T({2,3},{1,2,3,4,5,6})}}, {{"r", T({3,2},{1,4,2,5,3,6})}}},

    /* Whole-tensor reductions yield rank-1 [1], not a rank-0 scalar: CRPK
     * tensors always carry rank >= 1, so this is what the worker returns
     * (bark/kernel/ir_exec.c commits dims {1}). These goldens encoded the old
     * rank-0 shape, which is what made scalar_mul, concat and --verify each
     * need their own workaround. */
    {"reduce_sum", "calc::lambda_flow f(x) {\n x tensor_sum r =\n} return r;\n",
     {{"x", T({2,3},{1,2,3,4,5,6})}}, {{"r", T({1},{21})}}},

    {"reduce_mean", "calc::lambda_flow f(x) {\n x tensor_mean r =\n} return r;\n",
     {{"x", T({2,3},{1,2,3,4,5,6})}}, {{"r", T({1},{3})}}},

    {"reduce_max", "calc::lambda_flow f(x) {\n x tensor_max r =\n} return r;\n",
     {{"x", T({2,3},{1,2,3,4,5,6})}}, {{"r", T({1},{6})}}},

    {"reduce_min", "calc::lambda_flow f(x) {\n x tensor_min r =\n} return r;\n",
     {{"x", T({2,3},{1,2,3,4,5,6})}}, {{"r", T({1},{1})}}},

    {"clocked_parallel", "calc::lambda_flow f(a, b, c, d) {\n @clock(1):\n a b add r1 =\n c d mul r2 =\n @clock(2):\n r1 r2 add r =\n} return r;\n",
     {{"a", S(1)}, {"b", S(2)}, {"c", S(3)}, {"d", S(4)}}, {{"r", S(15)}}},  // (1+2)+(3*4)
  };

  for (const auto &c : cases) {
    ++g_cases;
    if (!run_case(c)) ++g_failures;
  }

  if (g_failures == 0) {
    std::printf("OK: all %d golden interpreter cases passed\n", g_cases);
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d/%d golden cases\n", g_failures, g_cases);
  return EXIT_FAILURE;
}
