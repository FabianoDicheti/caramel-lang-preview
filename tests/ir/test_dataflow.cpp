// ============================================================================
// Caramel Language - SSA dataflow graph tests
// ----------------------------------------------------------------------------
// Ticket: lang_014 (SSA/Dataflow Model)
// ============================================================================
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace caramel::parse;
using namespace caramel::ast;
using namespace caramel::ir;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

static const LambdaFlow *parse_flow(const std::string &src,
                                    std::unique_ptr<Program> &keep) {
  Lexer lx(src);
  Parser p(lx.tokenize());
  keep = p.parse();
  for (auto &it : keep->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      return static_cast<LambdaFlow *>(it.get());
  return nullptr;
}

static const DataflowNode *findOp(const DataflowGraph &g, const std::string &op) {
  for (const auto &n : g.nodes)
    if (n.kind == caramel::ir::NodeKind::Op && n.op == op) return &n;
  return nullptr;
}

// Clocked flow: two parallel ops at clock 1, a combiner at clock 2.
static void test_clocked_parallelism() {
  const char *src =
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "calc::lambda_flow f(a, b, c, d) {\n"
      "  @clock(1):\n"
      "    a b add r1 =\n"
      "    c d mul r2 =\n"
      "  @clock(2):\n"
      "    r1 r2 add result =\n"
      "} return result;\n";
  std::unique_ptr<Program> keep;
  const LambdaFlow *flow = parse_flow(src, keep);
  CHECK(flow != nullptr);
  DataflowGraph g = buildDataflow(*flow);

  // 4 params + add(r1) + mul(r2) + add(result) = 7 nodes
  CHECK(g.nodes.size() == 7);
  CHECK(g.outputs.size() == 1 && g.outputs[0] == "result");

  const DataflowNode *add1 = nullptr;
  const DataflowNode *mul = findOp(g, "mul");
  // there are two "add" ops; find the clock-1 one (operands are params)
  for (const auto &n : g.nodes)
    if (n.kind == caramel::ir::NodeKind::Op && n.op == "add" && n.level == 1)
      add1 = &n;
  CHECK(add1 != nullptr);
  CHECK(mul != nullptr);
  // clock(1): add and mul both at level 1 -> parallel
  CHECK(add1->level == 1);
  CHECK(mul->level == 1);
  // each consumes two param value ids
  CHECK(add1->operands.size() == 2);
  CHECK(mul->operands.size() == 2);
  // combiner at clock(2)
  const DataflowNode *combine = nullptr;
  for (const auto &n : g.nodes)
    if (n.kind == caramel::ir::NodeKind::Op && n.op == "add" && n.level == 2)
      combine = &n;
  CHECK(combine != nullptr);
  CHECK(combine->operands.size() == 2);

  // levels(): level 1 has two parallel ops
  auto lv = g.levels();
  CHECK(g.numLevels() == 3);  // 0 params, 1 parallel ops, 2 combine
  CHECK(lv[1].size() == 2);
  CHECK(lv[2].size() == 1);
}

// Unclocked flow: dependency-depth leveling. out = matmul(elemwise_add(x,y), W)
static void test_depth_leveling() {
  const char *src =
      "crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=1;\n"
      "calc::lambda_flow g(x, y, W) {\n"
      "  x y elemwise_add t =\n"
      "  t W matmul out =\n"
      "} return out;\n";
  std::unique_ptr<Program> keep;
  const LambdaFlow *flow = parse_flow(src, keep);
  CHECK(flow != nullptr);
  DataflowGraph g = buildDataflow(*flow);

  const DataflowNode *eadd = findOp(g, "elemwise_add");
  const DataflowNode *mm = findOp(g, "matmul");
  CHECK(eadd != nullptr && mm != nullptr);
  // params x,y,W at level 0; elemwise_add at 1; matmul depends on it -> level 2
  CHECK(eadd->level == 1);
  CHECK(mm->level == 2);
  // matmul consumes the elemwise_add result and the W param
  bool consumesEadd = false;
  for (ValueId v : mm->operands) if (v == eadd->id) consumesEadd = true;
  CHECK(consumesEadd);
  CHECK(g.outputs.size() == 1 && g.outputs[0] == "out");
}

// A tensor literal inside a flow body is a CONSTANT, not an external input.
// It used to fall through to the opaque-operand default and become an unbound
// `<expr>` Operand node, which either failed at run time with "missing input
// value" or produced a silently wrong result.
static void test_tensor_literal_lowers_to_const() {
  std::unique_ptr<Program> keep;
  const auto *flow = parse_flow(
      "crml::quantmax=100;\n"
      "crml::quantmin=-100;\n"
      "crml::quantres=0;\n"
      "calc::lambda_flow f(a) {\n"
      "    [10,20,30] a elemwise_add out =\n"
      "} return out;\n",
      keep);
  CHECK(flow != nullptr);
  if (!flow) return;
  auto g = buildDataflow(*flow);

  const DataflowNode *konst = nullptr;
  for (const auto &n : g.nodes) {
    CHECK(n.op != "<expr>");            // nothing opaque survives lowering
    if (n.kind == caramel::ir::NodeKind::Const) konst = &n;
  }
  CHECK(konst != nullptr);
  if (!konst) return;
  CHECK(konst->const_dims.size() == 1 && konst->const_dims[0] == 3);
  CHECK(konst->const_data == std::vector<int64_t>({10, 20, 30}));
}

// A nested (2-D) literal keeps row-major order and its full shape.
static void test_nested_tensor_literal_shape() {
  std::unique_ptr<Program> keep;
  const auto *flow = parse_flow(
      "crml::quantmax=100;\n"
      "crml::quantmin=-100;\n"
      "crml::quantres=0;\n"
      "calc::lambda_flow f(a) {\n"
      "    [[1,2,3],[4,5,6]] a matmul out =\n"
      "} return out;\n",
      keep);
  CHECK(flow != nullptr);
  if (!flow) return;
  auto g = buildDataflow(*flow);
  const DataflowNode *konst = nullptr;
  for (const auto &n : g.nodes)
    if (n.kind == caramel::ir::NodeKind::Const) konst = &n;
  CHECK(konst != nullptr);
  if (!konst) return;
  CHECK(konst->const_dims == std::vector<int64_t>({2, 3}));
  CHECK(konst->const_data == std::vector<int64_t>({1, 2, 3, 4, 5, 6}));
}

int main() {
  test_clocked_parallelism();
  test_depth_leveling();
  test_tensor_literal_lowers_to_const();
  test_nested_tensor_literal_shape();
  if (g_failures == 0) {
    std::printf("OK: all dataflow tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
