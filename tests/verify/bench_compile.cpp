// ============================================================================
// Objectives verification - compile-speed benchmark
// ----------------------------------------------------------------------------
// Objective: compile time < 1s for 1000 LOC  (README.md:441)
// Generates a ~1000-statement .crml flow and times the full front end
// (lex -> parse -> buildDataflow). Emits machine-parseable RESULT lines.
// ============================================================================
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <chrono>
#include <cstdio>
#include <string>

using namespace caramel::parse;

// Build a flow of `n` chained assignments (~n LOC of real RPN).
static std::string make_program(int n) {
  std::string s =
      "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
      "calc::lambda_flow big(a, b) {\n";
  s += "  a b add v0 =\n";
  for (int i = 1; i < n; ++i) {
    // alternate add/mul/sub so it isn't trivially foldable; chain prior result
    const char *op = (i % 3 == 0) ? "mul" : (i % 3 == 1) ? "add" : "sub";
    s += "  v" + std::to_string(i - 1) + " " + (i % 2 ? "a" : "b") + " " + op +
         " v" + std::to_string(i) + " =\n";
  }
  s += "} return v" + std::to_string(n - 1) + ";\n";
  return s;
}

int main() {
  const int kLines = 1000;
  std::string src = make_program(kLines);
  // count newlines as a LOC proxy
  int loc = 0;
  for (char c : src) if (c == '\n') ++loc;

  auto t0 = std::chrono::steady_clock::now();
  Lexer lx(src);
  auto toks = lx.tokenize();
  Parser p(std::move(toks));
  auto prog = p.parse();
  const caramel::ast::LambdaFlow *flow = nullptr;
  for (auto &it : prog->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      flow = static_cast<caramel::ast::LambdaFlow *>(it.get());
  caramel::ir::DataflowGraph g;
  if (flow) g = caramel::ir::buildDataflow(*flow);
  auto t1 = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double ms_per_kloc = ms * 1000.0 / loc;
  bool ok = p.ok() && flow != nullptr && ms < 1000.0;

  std::printf("RESULT compile_loc MEASURE %d\n", loc);
  std::printf("RESULT compile_dataflow_nodes MEASURE %zu\n", g.nodes.size());
  std::printf("RESULT compile_time_ms MEASURE %.3f\n", ms);
  std::printf("RESULT compile_ms_per_1000loc MEASURE %.3f\n", ms_per_kloc);
  std::printf("RESULT compile_under_1s_per_1000loc %s target=<1000ms measured=%.3fms\n",
              ok ? "PASS" : "FAIL", ms);
  return ok ? 0 : 1;
}
