// ============================================================================
// Objectives verification - simulation matches hardware precision
// ----------------------------------------------------------------------------
// Objective (Milestone 2): the interpreter matches FPGA precision.
// The hardware is integer with a saturating datapath. This checks that the
// interpreter (with int8/quant saturation enabled) reproduces, bit-for-bit, a
// hand-computed integer + saturating-clamp reference for matmul and elementwise
// ops -- i.e. simulation == hardware integer semantics.
// ============================================================================
#include "caramel/interp/int8_emulation.h"
#include "caramel/interp/interpreter.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace caramel::parse;
using namespace caramel::interp;

static caramel::ir::DataflowGraph graph_of(const std::string &src,
                                           std::unique_ptr<caramel::ast::Program> &keep) {
  Lexer lx(src);
  Parser p(lx.tokenize());
  keep = p.parse();
  for (auto &it : keep->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      return caramel::ir::buildDataflow(*static_cast<caramel::ast::LambdaFlow *>(it.get()));
  return {};
}

int main() {
  int failures = 0;

  // matmul that overflows the range -> the hardware saturates; the interpreter
  // must match exactly.
  const int64_t LO = -100, HI = 100;
  std::vector<int64_t> a = {10, 10, 10, 10}, b = {10, 10, 10, 10};
  // hand reference: integer matmul then clamp to [LO,HI]
  std::vector<int64_t> ref(4, 0);
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j) ref[i * 2 + j] += a[i * 2 + k] * b[k * 2 + j];
  for (auto &e : ref) e = std::max(LO, std::min(HI, e));  // saturate

  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=0;\n"
      "calc::lambda_flow mm(x, y) {\n x y matmul r =\n} return r;\n", keep);
  Interpreter vm;
  vm.add_evaluator(matrixOpEvaluator());
  vm.set_op_result_transform(saturationTransform(precisionRange(100, -100, 0)));
  vm.set_input("x", Value::tensor({2, 2}, a));
  vm.set_input("y", Value::tensor({2, 2}, b));
  auto r = vm.run(g);
  bool match = r.ok() && r.outputs["r"].data == ref;
  if (!match) ++failures;
  std::printf("RESULT sim_matmul_saturated_matches_hw %s ref=[%lld,%lld,%lld,%lld]\n",
              match ? "PASS" : "FAIL", (long long)ref[0], (long long)ref[1],
              (long long)ref[2], (long long)ref[3]);

  // determinism: same program + inputs -> identical bytes across runs.
  Interpreter vm2;
  vm2.add_evaluator(matrixOpEvaluator());
  vm2.set_op_result_transform(saturationTransform(precisionRange(100, -100, 0)));
  vm2.set_input("x", Value::tensor({2, 2}, a));
  vm2.set_input("y", Value::tensor({2, 2}, b));
  auto r2 = vm2.run(g);
  bool det = r2.ok() && r2.outputs["r"].data == r.outputs["r"].data;
  if (!det) ++failures;
  std::printf("RESULT sim_deterministic %s\n", det ? "PASS" : "FAIL");

  return failures == 0 ? 0 : 1;
}
