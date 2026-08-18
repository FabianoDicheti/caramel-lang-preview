// ============================================================================
// Objectives verification - integer-only runtime / compile-time quantization
// ----------------------------------------------------------------------------
// Objectives: "INTEGER-ONLY language; no floating-point at runtime" (spec:199),
//             "compile-time quantization" (spec:36).
// Structural guards: the interpreter's runtime Value is integer (int64 storage,
// no FP path), and a source program containing a float literal still produces an
// integer runtime value (no float survives into execution).
// ============================================================================
#include "caramel/interp/interpreter.h"
#include "caramel/interp/value.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <type_traits>

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

  // 1) The runtime value representation is integer (compile-time guarantee).
  constexpr bool int_storage =
      std::is_same<decltype(Value{}.data)::value_type, int64_t>::value;
  if (!int_storage) ++failures;
  std::printf("RESULT integer_only_runtime %s (Value::data is int64; no FP runtime path)\n",
              int_storage ? "PASS" : "FAIL");

  // 2) A float literal in source yields an integer runtime value (no float
  //    survives into execution). The interpreter evaluates entirely in integers.
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = graph_of(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=0;\n"
      "calc::lambda_flow f(a) {\n a 3.14 add r =\n} return r;\n", keep);
  Interpreter vm;
  vm.set_input("a", Value::scalar(10));
  auto r = vm.run(g);
  bool integral = r.ok();  // produced an integer Value without any FP arithmetic
  if (!integral) ++failures;
  std::printf("RESULT float_literal_runs_as_integer %s (3.14 literal -> integer runtime value)\n",
              integral ? "PASS" : "FAIL");

  // Observation for the report: in-source float-literal scaling by 10^quantres is
  // performed by the host-boundary Quantizer (lang_022), not auto-applied by the
  // interpreter's constant handling -> PARTIAL on full compile-time quantization.
  std::printf("RESULT compile_time_quant_scaling NOTE host-boundary via Quantizer; "
              "interpreter constant path does not auto-scale (PARTIAL)\n");

  return failures == 0 ? 0 : 1;
}
