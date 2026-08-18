// ============================================================================
// Caramel Language - end-to-end op-parameter pipeline test
// ----------------------------------------------------------------------------
// Ticket: x86_057/058/059 (IR v2 param channel -> frontend syntax)
// ----------------------------------------------------------------------------
// Proves `{ key: value }` object params flow all the way through:
//   parse -> buildDataflow (node.params + node.imm) -> serialize (IR immediate,
//   the worker path) AND -> interpret (the local --verify path), agreeing.
// The exemplar is softmax {axis:0}, whose expected values match the bark
// worker's 2x2 axis-0 vector in bark/tests/ir_exec_test.c.
// ============================================================================
#include "caramel/interp/activation.h"
#include "caramel/interp/interpreter.h"
#include "caramel/interp/norm.h"
#include "caramel/interp/shape.h"
#include "caramel/interp/spatial.h"
#include "caramel/interp/value.h"
#include "caramel/ir/binary_ir.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace caramel::parse;
using namespace caramel::ir;
using caramel::interp::Interpreter;
using caramel::interp::Value;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

static DataflowGraph build(const std::string &src,
                           std::unique_ptr<caramel::ast::Program> &keep) {
  Lexer lx(src);
  Parser p(lx.tokenize());
  keep = p.parse();
  for (auto &it : keep->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      return buildDataflow(*static_cast<caramel::ast::LambdaFlow *>(it.get()));
  return {};
}

int main() {
  // softmax with an explicit axis param, written as an object-params RPN term.
  const char *src =
      "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=3;\n"
      "calc::lambda_flow f(x) {\n"
      "  x { axis: 0 } softmax r =\n"
      "} return r;\n";
  std::unique_ptr<caramel::ast::Program> keep;
  DataflowGraph g = build(src, keep);

  // 1) The softmax node carries the resolved param and the folded immediate.
  const DataflowNode *sm = nullptr;
  for (const auto &n : g.nodes)
    if (n.kind == NodeKind::Op && n.op == "softmax") sm = &n;
  CHECK(sm != nullptr);
  if (sm) {
    bool has_axis0 = false;
    for (const auto &kv : sm->params)
      if (kv.first == "axis" && kv.second == 0) has_axis0 = true;
    CHECK(has_axis0);
    CHECK(sm->imm == 1u);  // axis 0 -> imm = axis + 1
  }

  // 2) Worker path: the serialized SOFTMAX instruction carries imm24 == 1.
  {
    auto bytes = serialize(g);
    auto mod = deserialize(bytes);
    CHECK(mod.has_value());
    const Instruction *si = nullptr;
    for (const auto &ins : mod->instructions)
      if (ins.opcode == OpCode::SOFTMAX) si = &ins;
    CHECK(si != nullptr);
    if (si) CHECK(instrImm24(*si) == 1u);
  }

  // 3) Local path: the interpreter honors axis 0 (columns normalized). Same
  //    2x2 input and result the worker asserts: cols [119,881].
  {
    Interpreter vm;
    vm.add_param_evaluator(caramel::interp::activationOpEvaluator(3));
    vm.set_input("x", Value::tensor({2, 2}, {1000, 2000, 3000, 4000}));
    auto res = vm.run(g);
    CHECK(res.ok());
    if (res.ok()) {
      auto it = res.outputs.find("r");
      CHECK(it != res.outputs.end());
      if (it != res.outputs.end())
        CHECK((it->second.data == std::vector<int64_t>{119, 119, 881, 881}));
    }
  }

  // 4) Pooling with two params { kernel_size, stride } -> packed imm, and the
  //    interpreter matches the worker's 4x4 k2s2 vectors.
  {
    const char *psrc =
        "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
        "calc::lambda_flow f(x) {\n"
        "  x { kernel_size: 2, stride: 2 } maxpool2d r =\n"
        "} return r;\n";
    std::unique_ptr<caramel::ast::Program> pk;
    DataflowGraph pg = build(psrc, pk);
    const DataflowNode *mp = nullptr;
    for (const auto &n : pg.nodes)
      if (n.kind == NodeKind::Op && n.op == "maxpool2d") mp = &n;
    CHECK(mp != nullptr);
    if (mp) CHECK(mp->imm == (2u | (2u << 8)));  // kernel 2 | stride 2

    auto mod = deserialize(serialize(pg));
    CHECK(mod.has_value());
    const Instruction *mi = nullptr;
    for (const auto &ins : mod->instructions)
      if (ins.opcode == OpCode::MAXPOOL2D) mi = &ins;
    CHECK(mi && instrImm24(*mi) == (2u | (2u << 8)));

    Interpreter vm;
    vm.add_param_evaluator(caramel::interp::spatialOpEvaluator());
    vm.set_input("x", Value::tensor({4, 4},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}));
    auto res = vm.run(pg);
    CHECK(res.ok());
    if (res.ok()) {
      auto it = res.outputs.find("r");
      CHECK(it != res.outputs.end());
      if (it != res.outputs.end())
        CHECK((it->second.data == std::vector<int64_t>{6, 8, 14, 16}));
    }
  }

  // 5) conv2d (2 operands + params) matches the worker's 3x3*diag2x2 vector.
  {
    Interpreter vm;
    vm.add_param_evaluator(caramel::interp::spatialOpEvaluator());
    // Direct evaluator call (no flow needed): x [3,3] * w [1,1,2,2] diag, s1p0.
    Value x = Value::tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Value w = Value::tensor({1, 1, 2, 2}, {1, 0, 0, 1});
    auto ev = caramel::interp::spatialOpEvaluator();
    std::string err;
    auto r = ev("conv2d", {x, w}, {{"stride", 1}, {"padding", 0}}, &err);
    CHECK(r.has_value());
    if (r) {
      CHECK((r->dims == std::vector<int64_t>{1, 2, 2}));
      CHECK((r->data == std::vector<int64_t>{6, 8, 12, 14}));
    }
  }

  // 6) concat along axis 1 matches the worker's 2x2||2x2 vector, and
  //    { axis: 1 } folds to imm 2.
  {
    const char *csrc =
        "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
        "calc::lambda_flow f(a, b) {\n"
        "  a b { axis: 1 } concat r =\n"
        "} return r;\n";
    std::unique_ptr<caramel::ast::Program> ck;
    DataflowGraph cg = build(csrc, ck);
    const DataflowNode *cn = nullptr;
    for (const auto &n : cg.nodes)
      if (n.kind == NodeKind::Op && n.op == "concat") cn = &n;
    CHECK(cn != nullptr);
    if (cn) CHECK(cn->imm == 2u);  // axis 1 -> imm = axis + 1

    Interpreter vm;
    vm.add_param_evaluator(caramel::interp::shapeOpEvaluator());
    vm.set_input("a", Value::tensor({2, 2}, {1, 2, 3, 4}));
    vm.set_input("b", Value::tensor({2, 2}, {5, 6, 7, 8}));
    auto res = vm.run(cg);
    CHECK(res.ok());
    if (res.ok()) {
      auto it = res.outputs.find("r");
      CHECK(it != res.outputs.end());
      if (it != res.outputs.end()) {
        CHECK((it->second.dims == std::vector<int64_t>{2, 4}));
        CHECK((it->second.data == std::vector<int64_t>{1, 2, 5, 6, 3, 4, 7, 8}));
      }
    }
  }

  // 7) layernorm over the last axis matches the worker's q1 vector.
  {
    const char *lsrc =
        "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=1;\n"
        "calc::lambda_flow f(x) {\n"
        "  x layernorm r =\n"
        "} return r;\n";
    std::unique_ptr<caramel::ast::Program> lk;
    DataflowGraph lg = build(lsrc, lk);
    // opcode reaches the stream
    auto mod = deserialize(serialize(lg));
    CHECK(mod.has_value());
    bool saw_ln = false;
    for (const auto &ins : mod->instructions)
      if (ins.opcode == OpCode::LAYERNORM) saw_ln = true;
    CHECK(saw_ln);

    Interpreter vm;
    vm.add_param_evaluator(caramel::interp::normOpEvaluator(1));  // q=1
    vm.set_input("x", Value::tensor({4}, {10, 20, 30, 40}));
    auto res = vm.run(lg);
    CHECK(res.ok());
    if (res.ok()) {
      auto it = res.outputs.find("r");
      CHECK(it != res.outputs.end());
      if (it != res.outputs.end())
        CHECK((it->second.data == std::vector<int64_t>{-13, -4, 4, 13}));
    }
  }

  if (g_failures == 0) {
    std::printf("OK: all param-op pipeline tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
