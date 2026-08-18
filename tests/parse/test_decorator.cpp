// ============================================================================
// Caramel Language - Decorator system tests
// ----------------------------------------------------------------------------
// Ticket: lang_006 (Decorator System)
// Build:  g++ -std=c++17 -I include tests/parse/test_decorator.cpp
//             src/parse/parser.cpp src/parse/lexer.cpp src/parse/decorator.cpp
//             src/ast/ast.cpp src/ops/op_registry.cpp src/types/type.cpp -o /tmp/td
// ============================================================================
#include "caramel/parse/decorator.h"
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

static std::unique_ptr<Program> parse_src(const std::string& src) {
  Lexer lx(src);
  Parser p(lx.tokenize());
  return p.parse();
}

static void test_registry() {
  const auto& reg = DecoratorRegistry::instance();
  CHECK(reg.size() >= 6);
  CHECK(reg.is_builtin("quantize"));
  CHECK(reg.is_builtin("tile"));
  CHECK(reg.is_builtin("device"));
  CHECK(reg.is_builtin("clock"));
  CHECK(!reg.is_builtin("not_a_decorator"));
  CHECK(reg.lookup("quantize")->kind == DecoratorKind::Quantization);
  CHECK(reg.lookup("device")->target == DecoratorTarget::Any);
  CHECK(reg.lookup("autodiff")->args_required == false);
}

static void test_inline_decorator_on_statement() {
  // @quantize{bits: 8} on a matmul assignment
  auto prog = parse_src(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "@quantize{bits: 8} x W matmul y =\n");
  const Assignment* a = nullptr;
  for (auto& it : prog->items)
    if (it->kind == NodeKind::Assignment) a = static_cast<Assignment*>(it.get());
  CHECK(a != nullptr);
  CHECK(a->decorators.size() == 1);
  CHECK(a->decorators[0].name == "quantize");
  CHECK(a->decorators[0].args != nullptr);
  CHECK(a->decorators[0].args->fields.size() == 1);
  CHECK(a->decorators[0].args->fields[0].key == "bits");
  // statement value still parses correctly
  CHECK(a->value && a->value->kind == NodeKind::OpApplication);
  CHECK(static_cast<OpApplication*>(a->value.get())->op == "matmul");
}

static void test_decorator_on_own_line_and_flow() {
  // @autodiff decorates the following lambda_flow; @tile decorates a statement.
  const char* src =
      "crml::quantres=2;\n"
      "crml::quantmax=100;\n"
      "crml::quantmin=-100;\n"
      "@autodiff\n"
      "calc::lambda_flow net(x, W) {\n"
      "  @tile{tile_m: 16, tile_n: 16}\n"
      "  x W matmul y =\n"
      "} return y;\n";
  auto prog = parse_src(src);
  const LambdaFlow* flow = nullptr;
  for (auto& it : prog->items)
    if (it->kind == NodeKind::LambdaFlow) flow = static_cast<LambdaFlow*>(it.get());
  CHECK(flow != nullptr);
  CHECK(flow->decorators.size() == 1);
  CHECK(flow->decorators[0].name == "autodiff");
  CHECK(flow->decorators[0].args == nullptr);  // no-arg decorator
  // the inner statement carries @tile
  CHECK(flow->body && !flow->body->is_clocked);
  CHECK(flow->body->statements.size() == 1);
  auto* st = static_cast<Assignment*>(flow->body->statements[0].get());
  CHECK(st->decorators.size() == 1);
  CHECK(st->decorators[0].name == "tile");
  CHECK(st->decorators[0].args && st->decorators[0].args->fields.size() == 2);
}

int main() {
  test_registry();
  test_inline_decorator_on_statement();
  test_decorator_on_own_line_and_flow();
  if (g_failures == 0) {
    std::printf("OK: all decorator tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
