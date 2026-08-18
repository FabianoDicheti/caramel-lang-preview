// ============================================================================
// Objectives verification - zero-overhead abstractions
// ----------------------------------------------------------------------------
// Objectives: "Object parameters compile to positional at compile-time" (spec:40),
//             stack ops dup/swap/drop are FREE / resolved (spec:736).
// Verifies the reduced SSA dataflow graph contains NO stack-op nodes (they are
// resolved during parsing) and that object-parameter literals do NOT appear as
// runtime data operands of the op they decorate.
// ============================================================================
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <set>
#include <string>

using namespace caramel::parse;

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

  // 1) Stack ops are resolved during reduction: none survive as dataflow nodes.
  std::unique_ptr<caramel::ast::Program> k1;
  auto g1 = graph_of(
      "crml::quantres=0;\n"
      "calc::lambda_flow res(x, W) {\n x dup W matmul swap elemwise_add r =\n} return r;\n",
      k1);
  const std::set<std::string> stackops = {"dup", "swap", "drop", "over", "rot", "-rot"};
  int stack_nodes = 0;
  for (const auto &n : g1.nodes)
    if (n.kind == caramel::ir::NodeKind::Op && stackops.count(n.op)) ++stack_nodes;
  bool no_stack = (stack_nodes == 0) && !g1.nodes.empty();
  if (!no_stack) ++failures;
  std::printf("RESULT zero_overhead_stackops %s stack_nodes_in_dataflow=%d (target=0)\n",
              no_stack ? "PASS" : "FAIL", stack_nodes);

  // 2) Object parameters do not become runtime data operands. conv2d with
  //    {stride,padding} keeps exactly its 2 tensor operands in the dataflow.
  std::unique_ptr<caramel::ast::Program> k2;
  auto g2 = graph_of(
      "crml::quantres=0;\n"
      "calc::lambda_flow c(image, filter) {\n"
      "  image filter {stride: 1, padding: 0} conv2d out =\n} return out;\n",
      k2);
  int conv_operands = -1;
  for (const auto &n : g2.nodes)
    if (n.kind == caramel::ir::NodeKind::Op && n.op == "conv2d")
      conv_operands = (int)n.operands.size();
  bool params_resolved = (conv_operands == 2);
  if (!params_resolved) ++failures;
  std::printf("RESULT zero_overhead_object_params %s conv2d_data_operands=%d (target=2)\n",
              params_resolved ? "PASS" : "FAIL", conv_operands);

  return failures == 0 ? 0 : 1;
}
