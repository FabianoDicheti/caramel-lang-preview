#include "caramel/lint/linter.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using caramel::lint::Diagnostic;
using caramel::parse::Lexer;
using caramel::parse::Parser;

static int failures = 0;
#define CHECK(condition)                                                   \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

static std::vector<Diagnostic> lintSource(const std::string& source) {
  Lexer lexer(source);
  auto tokens = lexer.tokenize();
  CHECK(lexer.errors().empty());
  Parser parser(std::move(tokens));
  auto program = parser.parse();
  CHECK(parser.errors().empty());
  return caramel::lint::lint(*program);
}

static bool hasCode(const std::vector<Diagnostic>& diagnostics,
                    const std::string& code) {
  for (const auto& diagnostic : diagnostics)
    if (diagnostic.code == code) return true;
  return false;
}

static int countCode(const std::vector<Diagnostic>& diagnostics,
                     const std::string& code) {
  int count = 0;
  for (const auto& diagnostic : diagnostics)
    if (diagnostic.code == code) ++count;
  return count;
}

static const char* preamble =
    "crml::quantmax=100;\n"
    "crml::quantmin=-100;\n"
    "crml::quantres=2;\n";

static void testCleanFlow() {
  auto diagnostics = lintSource(
      std::string(preamble) +
      "calc::lambda_flow add_values(a, b) {\n"
      "  a b add result =\n"
      "} return result;\n");
  CHECK(diagnostics.empty());
}

static void testUndefinedAndUnused() {
  auto diagnostics = lintSource(
      std::string(preamble) +
      "calc::lambda_flow broken(a, spare) {\n"
      "  a missing add result =\n"
      "  1 dead =\n"
      "} return result;\n");
  CHECK(hasCode(diagnostics, "E003"));
  CHECK(hasCode(diagnostics, "W001"));
  CHECK(hasCode(diagnostics, "W004"));
}

static void testDuplicateDefinitionsAndParameters() {
  auto diagnostics = lintSource(
      std::string(preamble) +
      "calc::lambda_flow same(x, x) { x out = } return out;\n"
      "calc::lambda_flow same(y) { y out = } return out;\n");
  CHECK(hasCode(diagnostics, "E001"));
  CHECK(hasCode(diagnostics, "E004"));
}

static void testAssignmentsAndReturn() {
  auto diagnostics = lintSource(
      std::string(preamble) +
      "calc::lambda_flow f(x) {\n"
      "  x value =\n"
      "  x value =\n"
      "  x pair, pair =\n"
      "} return absent;\n");
  CHECK(hasCode(diagnostics, "W002"));
  CHECK(hasCode(diagnostics, "E002"));
  CHECK(hasCode(diagnostics, "E005"));
}

static void testObjectParameters() {
  auto diagnostics = lintSource(
      std::string(preamble) +
      "calc::lambda_flow f(image, filter) {\n"
      "  image filter {stride:1, stride:2} conv2d out =\n"
      "} return out;\n");
  CHECK(countCode(diagnostics, "W003") == 1);
}

static void testInputIsGlobal() {
  auto diagnostics = lintSource(
      std::string(preamble) +
      "in::x = 5;\n"
      "calc::lambda_flow f() { x out = } return out;\n");
  CHECK(!hasCode(diagnostics, "E003"));
}

int main() {
  testCleanFlow();
  testUndefinedAndUnused();
  testDuplicateDefinitionsAndParameters();
  testAssignmentsAndReturn();
  testObjectParameters();
  testInputIsGlobal();
  if (failures == 0) std::puts("all linter tests passed");
  return failures == 0 ? 0 : 1;
}
