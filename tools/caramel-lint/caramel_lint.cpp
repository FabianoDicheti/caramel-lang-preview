#include "caramel/lint/linter.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void printDiagnostic(const std::string& path, const char* severity,
                     const std::string& code, const std::string& message,
                     const caramel::ast::SourceLocation& loc) {
  std::cerr << path << ':' << loc.line << ':' << loc.column << ": " << severity;
  if (!code.empty()) std::cerr << '[' << code << ']';
  std::cerr << ": " << message << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: caramel-lint <file.crml>\n";
    return 2;
  }

  const std::string path = argv[1];
  std::ifstream input(path);
  if (!input) {
    std::cerr << "caramel-lint: cannot open " << path << '\n';
    return 2;
  }
  std::stringstream buffer;
  buffer << input.rdbuf();

  caramel::parse::Lexer lexer(buffer.str());
  auto tokens = lexer.tokenize();
  bool hasError = false;
  for (const auto& error : lexer.errors()) {
    printDiagnostic(path, "error", "ELEX", error.message, error.loc);
    hasError = true;
  }

  caramel::parse::Parser parser(std::move(tokens));
  auto program = parser.parse();
  for (const auto& error : parser.errors()) {
    printDiagnostic(path, "error", "EPARSE", error.message, error.loc);
    hasError = true;
  }
  for (const auto& warning : parser.warnings())
    printDiagnostic(path, "warning", "WPARSE", warning.message, warning.loc);

  // Semantic rules only run for syntactically valid programs, avoiding noisy
  // follow-on errors from the parser's intentionally partial recovery AST.
  if (!hasError) {
    for (const auto& diagnostic : caramel::lint::lint(*program)) {
      printDiagnostic(path, caramel::lint::severityName(diagnostic.severity),
                      diagnostic.code, diagnostic.message, diagnostic.loc);
      hasError |= diagnostic.severity == caramel::lint::Severity::Error;
    }
  }
  return hasError ? 1 : 0;
}
