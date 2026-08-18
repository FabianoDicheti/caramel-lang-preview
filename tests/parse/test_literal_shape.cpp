// ============================================================================
// Caramel Language - Matrix literal shape inference tests
// ----------------------------------------------------------------------------
// Ticket: lang_007 (Matrix Initialization Syntax)
// ============================================================================
#include "caramel/parse/lexer.h"
#include "caramel/parse/literal_shape.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>

using namespace caramel::parse;
using namespace caramel::ast;
using caramel::types::Shape;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// Build a tensor literal by parsing an assignment and grabbing its value.
static const TensorLiteral *parse_literal(const std::string &rhs,
                                          std::unique_ptr<Program> &keep,
                                          std::vector<ParseError> *errs = nullptr) {
  Lexer lx("crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=0;\n" +
           rhs + " m =\n");
  Parser p(lx.tokenize());
  keep = p.parse();
  if (errs) *errs = p.errors();
  for (auto &it : keep->items)
    if (it->kind == NodeKind::Assignment) {
      auto *a = static_cast<Assignment *>(it.get());
      if (a->value && a->value->kind == NodeKind::TensorLiteral)
        return static_cast<TensorLiteral *>(a->value.get());
    }
  return nullptr;
}

static void test_2d_shape() {
  std::unique_ptr<Program> keep;
  const TensorLiteral *lit = parse_literal("[[1, 2, 3], [4, 5, 6]]", keep);
  CHECK(lit != nullptr);
  auto s = inferTensorLiteralShape(*lit);
  CHECK(s.has_value());
  CHECK((s->dims == std::vector<int64_t>{2, 3}));
  CHECK(isRectangular(*lit));
}

static void test_3d_shape() {
  std::unique_ptr<Program> keep;
  const TensorLiteral *lit =
      parse_literal("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", keep);
  CHECK(lit != nullptr);
  auto s = inferTensorLiteralShape(*lit);
  CHECK(s.has_value());
  CHECK((s->dims == std::vector<int64_t>{2, 2, 2}));
}

static void test_1d_shape() {
  std::unique_ptr<Program> keep;
  const TensorLiteral *lit = parse_literal("[1, 2, 3, 4, 5]", keep);
  CHECK(lit != nullptr);
  auto s = inferTensorLiteralShape(*lit);
  CHECK(s.has_value());
  CHECK((s->dims == std::vector<int64_t>{5}));
}

static void test_ragged_is_rejected() {
  // Ragged rows: [[1,2,3],[4,5]] -> the parser should record a diagnostic.
  std::unique_ptr<Program> keep;
  std::vector<ParseError> errs;
  const TensorLiteral *lit = parse_literal("[[1, 2, 3], [4, 5]]", keep, &errs);
  CHECK(lit != nullptr);
  std::string err;
  CHECK(!inferTensorLiteralShape(*lit, &err).has_value());
  CHECK(!err.empty());
  CHECK(!isRectangular(*lit));
  // the parser surfaced a dimension-mismatch diagnostic
  CHECK(!errs.empty());
}

static void test_mixed_levels_rejected() {
  // Mixed scalar + nested at the same level: [1, [2, 3]] -> ragged.
  std::unique_ptr<Program> keep;
  const TensorLiteral *lit = parse_literal("[1, [2, 3]]", keep);
  CHECK(lit != nullptr);
  CHECK(!isRectangular(*lit));
}

int main() {
  test_2d_shape();
  test_3d_shape();
  test_1d_shape();
  test_ragged_is_rejected();
  test_mixed_levels_rejected();
  if (g_failures == 0) {
    std::printf("OK: all matrix-literal shape tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
