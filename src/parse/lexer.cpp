// ============================================================================
// Caramel Language - Lexer implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_005 (RPN Parser Implementation)
// Version: 1.0.0
// ============================================================================
#include "caramel/parse/lexer.h"

#include <cctype>

namespace caramel::parse {

using caramel::ast::SourceLocation;

const char* tok_kind_name(TokKind k) {
  switch (k) {
    case TokKind::Identifier: return "Identifier";
    case TokKind::Number:     return "Number";
    case TokKind::String:     return "String";
    case TokKind::ColonColon: return "ColonColon";
    case TokKind::Arrow:      return "Arrow";
    case TokKind::Compare:    return "Compare";
    case TokKind::LBrace:     return "LBrace";
    case TokKind::RBrace:     return "RBrace";
    case TokKind::LBracket:   return "LBracket";
    case TokKind::RBracket:   return "RBracket";
    case TokKind::LParen:     return "LParen";
    case TokKind::RParen:     return "RParen";
    case TokKind::Colon:      return "Colon";
    case TokKind::Comma:      return "Comma";
    case TokKind::Semicolon:  return "Semicolon";
    case TokKind::Dot:        return "Dot";
    case TokKind::Assign:     return "Assign";
    case TokKind::Newline:    return "Newline";
    case TokKind::EndOfFile:  return "EndOfFile";
    case TokKind::Invalid:    return "Invalid";
  }
  return "Invalid";
}

bool Lexer::at_end() const { return pos_ >= src_.size(); }

char Lexer::peek(int ahead) const {
  std::size_t p = pos_ + static_cast<std::size_t>(ahead);
  return p < src_.size() ? src_[p] : '\0';
}

char Lexer::advance() {
  char c = src_[pos_++];
  if (c == '\n') { ++line_; col_ = 1; } else { ++col_; }
  return c;
}

void Lexer::add_error(const std::string& msg) {
  errors_.push_back({msg, SourceLocation{line_, col_, static_cast<uint32_t>(pos_)}});
}

static bool is_ident_start(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
static bool is_ident_cont(char c) { return std::isalnum((unsigned char)c) || c == '_'; }

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> out;
  auto loc_here = [&]() {
    return SourceLocation{line_, col_, static_cast<uint32_t>(pos_)};
  };

  while (!at_end()) {
    char c = peek();

    // Whitespace (not newline)
    if (c == ' ' || c == '\t' || c == '\r') { advance(); continue; }

    // Comment to end of line
    if (c == '#') {
      while (!at_end() && peek() != '\n') advance();
      continue;
    }

    // Newline (statement terminator)
    if (c == '\n') {
      SourceLocation l = loc_here();
      advance();
      out.push_back({TokKind::Newline, "\\n", l, false});
      continue;
    }

    SourceLocation start = loc_here();

    // Numeric literal: digit, or '-' immediately followed by a digit ([A-10]).
    if (std::isdigit((unsigned char)c) ||
        (c == '-' && std::isdigit((unsigned char)peek(1)))) {
      std::string num;
      if (c == '-') num.push_back(advance());
      bool is_float = false;
      while (std::isdigit((unsigned char)peek())) num.push_back(advance());
      if (peek() == '.' && std::isdigit((unsigned char)peek(1))) {
        is_float = true;
        num.push_back(advance());  // '.'
        while (std::isdigit((unsigned char)peek())) num.push_back(advance());
      }
      out.push_back({TokKind::Number, num, start, is_float});
      continue;
    }

    // Identifier (including the @clock marker, lexed as one identifier "@clock",
    // and the reserved stack op "-rot").
    if (is_ident_start(c) || (c == '@' && is_ident_start(peek(1))) ||
        (c == '-' && is_ident_start(peek(1)))) {
      std::string id;
      if (c == '@' || c == '-') id.push_back(advance());
      while (is_ident_cont(peek())) id.push_back(advance());
      out.push_back({TokKind::Identifier, id, start, false});
      continue;
    }

    // String literal (double or single quoted; no escapes in v1.0)
    if (c == '"' || c == '\'') {
      char quote = advance();
      std::string s;
      while (!at_end() && peek() != quote && peek() != '\n') s.push_back(advance());
      if (peek() != quote) {
        add_error("unterminated string literal");
      } else {
        advance();  // closing quote
      }
      Token t{TokKind::String, s, start, false};
      out.push_back(t);
      continue;
    }

    // Multi-char operators
    if (c == ':' && peek(1) == ':') { advance(); advance(); out.push_back({TokKind::ColonColon, "::", start, false}); continue; }
    if (c == '=' && peek(1) == '>') { advance(); advance(); out.push_back({TokKind::Arrow, "=>", start, false}); continue; }
    if (c == '=' && peek(1) == '=') { advance(); advance(); out.push_back({TokKind::Compare, "==", start, false}); continue; }
    if (c == '>' && peek(1) == '=') { advance(); advance(); out.push_back({TokKind::Compare, ">=", start, false}); continue; }
    if (c == '<' && peek(1) == '=') { advance(); advance(); out.push_back({TokKind::Compare, "<=", start, false}); continue; }
    if (c == '!' && peek(1) == '=') { advance(); advance(); out.push_back({TokKind::Compare, "!=", start, false}); continue; }

    // Single-char tokens
    advance();
    switch (c) {
      case '{': out.push_back({TokKind::LBrace, "{", start, false}); break;
      case '}': out.push_back({TokKind::RBrace, "}", start, false}); break;
      case '[': out.push_back({TokKind::LBracket, "[", start, false}); break;
      case ']': out.push_back({TokKind::RBracket, "]", start, false}); break;
      case '(': out.push_back({TokKind::LParen, "(", start, false}); break;
      case ')': out.push_back({TokKind::RParen, ")", start, false}); break;
      case ':': out.push_back({TokKind::Colon, ":", start, false}); break;
      case ',': out.push_back({TokKind::Comma, ",", start, false}); break;
      case ';': out.push_back({TokKind::Semicolon, ";", start, false}); break;
      case '.': out.push_back({TokKind::Dot, ".", start, false}); break;
      case '=': out.push_back({TokKind::Assign, "=", start, false}); break;
      case '>': out.push_back({TokKind::Compare, ">", start, false}); break;
      case '<': out.push_back({TokKind::Compare, "<", start, false}); break;
      default:
        add_error(std::string("unexpected character '") + c + "'");
        out.push_back({TokKind::Invalid, std::string(1, c), start, false});
        break;
    }
  }

  out.push_back({TokKind::EndOfFile, "", SourceLocation{line_, col_, static_cast<uint32_t>(pos_)}, false});
  return out;
}

}  // namespace caramel::parse
