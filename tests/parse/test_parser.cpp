// ============================================================================
// Caramel Language - Parser tests (end-to-end: source -> tokens -> AST)
// ----------------------------------------------------------------------------
// Ticket: lang_005 (RPN Parser Implementation)
// Build:  g++ -std=c++17 -I include tests/parse/test_parser.cpp
//             src/parse/parser.cpp src/parse/lexer.cpp src/ast/ast.cpp
//             src/ops/op_registry.cpp src/types/type.cpp -o /tmp/test_parser
// ============================================================================
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>

using namespace caramel::parse;
using namespace caramel::ast;

static int g_failures = 0;
#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
      ++g_failures;                                                  \
    }                                                                \
  } while (0)

static std::unique_ptr<Program> parse_src(const std::string& src,
                                          std::vector<ParseError>* errs = nullptr) {
  Lexer lx(src);
  auto toks = lx.tokenize();
  Parser p(std::move(toks));
  auto prog = p.parse();
  if (errs) *errs = p.errors();
  return prog;
}

// Reduce a single RPN assignment and return its value expression.
static const Assignment* first_assignment(const Program& prog) {
  for (const auto& item : prog.items)
    if (item->kind == NodeKind::Assignment)
      return static_cast<const Assignment*>(item.get());
  return nullptr;
}

static void test_simple_assignment() {
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "5 3 add result =\n");
  // directives (3) + assignment (1)
  int dirs = 0, asn = 0;
  for (auto& it : prog->items) {
    if (it->kind == NodeKind::Directive) ++dirs;
    if (it->kind == NodeKind::Assignment) ++asn;
  }
  CHECK(dirs == 3);
  CHECK(asn == 1);
  const Assignment* a = first_assignment(*prog);
  CHECK(a != nullptr);
  CHECK(a->targets.size() == 1 && a->targets[0] == "result");
  // value is OpApplication add(5, 3)
  CHECK(a->value && a->value->kind == NodeKind::OpApplication);
  auto* op = static_cast<OpApplication*>(a->value.get());
  CHECK(op->op == "add");
  CHECK(op->args.size() == 2);
  CHECK(op->args[0]->kind == NodeKind::NumberLiteral);
  CHECK(static_cast<NumberLiteral*>(op->args[0].get())->lexeme == "5");
}

static void test_nested_and_type() {
  // x y elemwise_add W matmul out oc::tensor =
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "x y elemwise_add W matmul out oc::tensor =\n");
  const Assignment* a = first_assignment(*prog);
  CHECK(a != nullptr);
  CHECK(a->targets.size() == 1 && a->targets[0] == "out");
  CHECK(a->type.has_value());
  CHECK(a->type->name == "tensor");
  // value: matmul(elemwise_add(x,y), W)
  CHECK(a->value->kind == NodeKind::OpApplication);
  auto* mm = static_cast<OpApplication*>(a->value.get());
  CHECK(mm->op == "matmul");
  CHECK(mm->args.size() == 2);
  CHECK(mm->args[0]->kind == NodeKind::OpApplication);
  CHECK(static_cast<OpApplication*>(mm->args[0].get())->op == "elemwise_add");
  CHECK(mm->args[1]->kind == NodeKind::VarRef);
  CHECK(static_cast<VarRef*>(mm->args[1].get())->name == "W");
}

static void test_object_params_and_qualified() {
  // object params attach to the following op; qualified refs parse.
  auto prog = parse_src(
      "crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=0;\n"
      "image filter {stride: 1, padding: 0} conv2d out =\n"
      "port::0 receiver::matrix_a =\n");
  // find the conv2d assignment
  const OpApplication* conv = nullptr;
  const Assignment* bind = nullptr;  // port::0 receiver::matrix_a =
  for (auto& it : prog->items) {
    if (it->kind != NodeKind::Assignment) continue;
    auto* a = static_cast<Assignment*>(it.get());
    if (a->value && a->value->kind == NodeKind::OpApplication) {
      auto* o = static_cast<OpApplication*>(a->value.get());
      if (o->op == "conv2d") conv = o;
    }
    if (a->value && a->value->kind == NodeKind::QualifiedRef) bind = a;
  }
  CHECK(conv != nullptr);
  if (conv) {
    // args: image, filter, + trailing ObjectParams (params retained in tree)
    CHECK(conv->args.size() == 3);
    CHECK(conv->args.back()->kind == NodeKind::ObjectParams);
    auto* op = static_cast<ObjectParams*>(conv->args.back().get());
    CHECK(op->fields.size() == 2 && op->fields[0].key == "stride");
  }
  // value is port::0; target is the qualified name receiver::matrix_a
  CHECK(bind != nullptr);
  if (bind) {
    auto* qr = static_cast<QualifiedRef*>(bind->value.get());
    CHECK(qr->ns == "port" && qr->member == "0");
    CHECK(bind->targets.size() == 1 && bind->targets[0] == "receiver::matrix_a");
  }
}

static void test_lambda_flow_and_clocks() {
  const char* src =
      "crml::quantres=2;\n"
      "crml::quantmax=100;\n"
      "crml::quantmin=-100;\n"
      "calc::lambda_flow f(a, b, c, d) {\n"
      "  @clock(1):\n"
      "    a b add r1 =\n"
      "    c d mul r2 =\n"
      "  @clock(2):\n"
      "    r1 r2 add result =\n"
      "} return result;\n"
      "calc::lambda_calculator(channels=3) {\n"
      "  port::0 receiver::x =\n"
      "  receiver::x receiver::x receiver::x receiver::x f out =\n"
      "  out port::2 responser::result =\n"
      "}\n";
  std::vector<ParseError> errs;
  auto prog = parse_src(src, &errs);
  CHECK(errs.empty());
  const LambdaFlow* flow = nullptr;
  const LambdaCalculator* calc = nullptr;
  for (auto& it : prog->items) {
    if (it->kind == NodeKind::LambdaFlow) flow = static_cast<LambdaFlow*>(it.get());
    if (it->kind == NodeKind::LambdaCalculator) calc = static_cast<LambdaCalculator*>(it.get());
  }
  CHECK(flow != nullptr);
  CHECK(flow->name == "f");
  CHECK(flow->params.size() == 4);
  CHECK(flow->returns.size() == 1 && flow->returns[0] == "result");
  CHECK(flow->body && flow->body->is_clocked);
  CHECK(flow->body->clocks.size() == 2);
  CHECK(flow->body->clocks[0]->level == 1);
  CHECK(flow->body->clocks[0]->statements.size() == 2);
  CHECK(flow->body->clocks[1]->statements.size() == 1);
  CHECK(calc != nullptr);
  CHECK(calc->channels.has_value() && *calc->channels == 3);
  CHECK(calc->body && !calc->body->is_clocked);
  CHECK(calc->body->statements.size() == 3);
}

static void test_dup_swap() {
  // residual: x dup W matmul swap elemwise_add out =
  // -> elemwise_add(matmul(x, W), x)
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "x dup W matmul swap elemwise_add out =\n");
  const Assignment* a = first_assignment(*prog);
  CHECK(a != nullptr);
  CHECK(a->value->kind == NodeKind::OpApplication);
  auto* add = static_cast<OpApplication*>(a->value.get());
  CHECK(add->op == "elemwise_add");
  CHECK(add->args.size() == 2);
  CHECK(add->args[0]->kind == NodeKind::OpApplication);
  CHECK(static_cast<OpApplication*>(add->args[0].get())->op == "matmul");
  CHECK(add->args[1]->kind == NodeKind::VarRef);  // the dup'd x
  CHECK(static_cast<VarRef*>(add->args[1].get())->name == "x");
}

static void test_tuple_targets() {
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "a b train_step updated_params, loss =\n");
  // train_step is unknown (no user fn / not primitive) -> treated as VarRef,
  // so reduction leaves the last operand; targets must still split on comma.
  const Assignment* a = first_assignment(*prog);
  CHECK(a != nullptr);
  CHECK(a->targets.size() == 2);
  CHECK(a->targets[0] == "updated_params");
  CHECK(a->targets[1] == "loss");
}

static void test_parser_contract_errors() {
  std::vector<ParseError> errs;
  parse_src("5 3 add result =\n", &errs);
  CHECK(!errs.empty());

  errs.clear();
  parse_src("crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=0;\n5 add =\n", &errs);
  CHECK(!errs.empty());

  errs.clear();
  parse_src("crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=0;\n5 add out =\n", &errs);
  CHECK(!errs.empty());
}

static void test_minus_rot_and_reserved_ops() {
  std::vector<ParseError> errs;
  auto prog = parse_src(
      "crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=0;\n"
      "a b c -rot add out =\n"
      "Q K V {num_heads: 8, embed_dim: 64} multihead_attention mha =\n",
      &errs);
  CHECK(errs.empty());
  int assignments = 0;
  for (auto& it : prog->items) {
    if (it->kind != NodeKind::Assignment) continue;
    ++assignments;
    auto* a = static_cast<Assignment*>(it.get());
    CHECK(a->value && a->value->kind == NodeKind::OpApplication);
  }
  CHECK(assignments == 2);
}

static void test_mixed_clock_block() {
  std::vector<ParseError> errs;
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "calc::lambda_flow f(a, b) {\n"
      "  a b add pre =\n"
      "  @clock(1):\n"
      "    pre b mul mid =\n"
      "  mid b add result =\n"
      "} return result;\n",
      &errs);
  CHECK(errs.empty());
  const LambdaFlow* flow = nullptr;
  for (auto& it : prog->items)
    if (it->kind == NodeKind::LambdaFlow) flow = static_cast<LambdaFlow*>(it.get());
  CHECK(flow != nullptr);
  CHECK(flow->body && flow->body->is_clocked);
  CHECK(flow->body->statements.size() == 2);
  CHECK(flow->body->clocks.size() == 1);
}

static void test_status_directive() {
  std::vector<ParseError> errs;
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "status::bob;\n",
      &errs);
  CHECK(errs.empty());
  const Directive* status = nullptr;
  for (auto& it : prog->items)
    if (it->kind == NodeKind::Directive) {
      auto* d = static_cast<Directive*>(it.get());
      if (d->dkind == DirectiveKind::Status) status = d;
    }
  CHECK(status != nullptr);
  CHECK(status && status->key == "bob");
}

static void test_status_directive_requires_alias() {
  std::vector<ParseError> errs;
  parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "status::;\n",
      &errs);
  CHECK(!errs.empty());
}

static void test_profile_directive() {
  std::vector<ParseError> errs;
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "in::a = [[1,2],[3,4]];\n"
      "profile::a;\n",
      &errs);
  CHECK(errs.empty());
  const Directive* prof = nullptr;
  for (auto& it : prog->items)
    if (it->kind == NodeKind::Directive) {
      auto* d = static_cast<Directive*>(it.get());
      if (d->dkind == DirectiveKind::Profile) prof = d;
    }
  CHECK(prof != nullptr);
  CHECK(prof && prof->key == "a");
}

int main() {
  test_simple_assignment();
  test_nested_and_type();
  test_object_params_and_qualified();
  test_lambda_flow_and_clocks();
  test_dup_swap();
  test_tuple_targets();
  test_parser_contract_errors();
  test_minus_rot_and_reserved_ops();
  test_mixed_clock_block();
  test_status_directive();
  test_status_directive_requires_alias();
  test_profile_directive();
  // A return list that wraps across lines must keep every name. The parser
  // used to stop at the first Newline and silently drop the rest: a flow
  // returning 19 values over three lines yielded the first seven and reported
  // success. A list of results is the worst place for a silent truncation.
  {
    const char* src =
        "calc::lambda_flow g(a, b) {\n"
        "    a b elemwise_add r1 =\n"
        "    a b elemwise_sub r2 =\n"
        "    a b elemwise_mul r3 =\n"
        "} return r1, r2,\n"
        "         r3;\n";
    auto p2 = parse_src(src);
    const LambdaFlow* g = nullptr;
    if (p2) {
      for (auto& it : p2->items)
        if (it->kind == NodeKind::LambdaFlow) g = static_cast<LambdaFlow*>(it.get());
    }
    CHECK(g != nullptr);
    CHECK(g && g->returns.size() == 3);
    CHECK(g && g->returns.size() == 3 && g->returns[0] == "r1" &&
          g->returns[1] == "r2" && g->returns[2] == "r3");
  }

  // ...but a newline still ends a list that did not end on a comma, so a
  // missing semicolon cannot swallow whatever follows.
  {
    const char* src =
        "calc::lambda_flow h(a) {\n"
        "    a relu r1 =\n"
        "} return r1\n"
        "calc::lambda_flow h2(a) {\n"
        "    a relu r9 =\n"
        "} return r9;\n";
    auto p3 = parse_src(src);
    const LambdaFlow* first = nullptr;
    if (p3) {
      for (auto& it : p3->items) {
        if (it->kind == NodeKind::LambdaFlow) { first = static_cast<LambdaFlow*>(it.get()); break; }
      }
    }
    CHECK(first != nullptr);
    CHECK(first && first->returns.size() == 1 && first->returns[0] == "r1");
  }

  if (g_failures == 0) {
    std::printf("OK: all parser tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
