// ============================================================================
// Caramel Language - Lexer
// ----------------------------------------------------------------------------
// Ticket:   lang_005 (RPN Parser Implementation)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// O(n) tokenizer for the grammar in grammar/caramel.ebnf. Produces a flat token
// stream the RPN parser reduces with an operand stack. Whitespace separates
// tokens; comments (# ... EOL) are dropped; NEWLINE is emitted as a token because
// it is a statement terminator ([A-4]). Multi-char tokens recognized: "::",
// "=>", and the comparison operators (==, >=, <=, !=, >, <). A leading '-'
// directly before a digit is part of a numeric literal ([A-10]).
// ============================================================================
#ifndef CARAMEL_PARSE_LEXER_H
#define CARAMEL_PARSE_LEXER_H

#include <string>
#include <vector>

#include "caramel/ast/ast.h"  // SourceLocation

namespace caramel::parse {

enum class TokKind {
  Identifier,   // foo, matrix_a, add, relu (reserved-vs-user decided by parser)
  Number,       // 5, -75.5, 3.14
  String,       // "..." or '...'
  ColonColon,   // ::
  Arrow,        // =>
  Compare,      // > < == >= <= !=   (text in `text`)
  LBrace, RBrace, LBracket, RBracket, LParen, RParen,
  Colon, Comma, Semicolon, Dot, Assign,  // : , ; . =
  Newline,
  EndOfFile,
  Invalid,
};

struct Token {
  TokKind kind = TokKind::Invalid;
  std::string text;                 // lexeme
  caramel::ast::SourceLocation loc; // start location
  bool is_float = false;            // for Number
};

struct LexError {
  std::string message;
  caramel::ast::SourceLocation loc;
};

class Lexer {
 public:
  explicit Lexer(std::string source) : src_(std::move(source)) {}

  // Tokenize the whole buffer. Always ends with an EndOfFile token. Lexical
  // errors are accumulated in errors().
  std::vector<Token> tokenize();

  const std::vector<LexError>& errors() const { return errors_; }
  bool ok() const { return errors_.empty(); }

 private:
  char peek(int ahead = 0) const;
  char advance();
  bool at_end() const;
  void add_error(const std::string& msg);

  std::string src_;
  std::size_t pos_ = 0;
  uint32_t line_ = 1;
  uint32_t col_ = 1;
  std::vector<LexError> errors_;
};

const char* tok_kind_name(TokKind k);

}  // namespace caramel::parse

#endif  // CARAMEL_PARSE_LEXER_H
