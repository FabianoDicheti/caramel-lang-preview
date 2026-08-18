#ifndef CARAMEL_LINT_LINTER_H
#define CARAMEL_LINT_LINTER_H

#include <string>
#include <vector>

#include "caramel/ast/ast.h"

namespace caramel::lint {

enum class Severity { Warning, Error };

struct Diagnostic {
  Severity severity = Severity::Warning;
  std::string code;
  std::string message;
  ast::SourceLocation loc{};
};

// Run semantic lint rules over an already parsed program. Syntax diagnostics
// remain the responsibility of the lexer/parser so this pass can assume a
// structurally valid AST.
std::vector<Diagnostic> lint(const ast::Program& program);

const char* severityName(Severity severity);

}  // namespace caramel::lint

#endif  // CARAMEL_LINT_LINTER_H
