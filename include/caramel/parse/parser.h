// ============================================================================
// Caramel Language - RPN Parser
// ----------------------------------------------------------------------------
// Ticket:   lang_005 (RPN Parser Implementation)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Single-pass, stack-based parser. Consumes the lexer's token stream and the
// grammar (lang_001), builds the AST (lang_002), and uses operator arities from
// the op registry (lang_004) plus a table of user lambda_flow/symphony arities to
// reduce RPN expressions. No backtracking (O(n)).
//
// v1.0 coverage: quantization/directive preamble; top-level assignments and
// print statements; calc::lambda_flow and calc::lambda_calculator definitions
// with (optionally @clock-partitioned) bodies of assignments / prints /
// memory writes; full RPN expression reduction (literals, qualified/indexed/
// property refs, object parameters, primitive & user-function applications,
// stack ops, combinators) with the [A-5] target rule.
// Deferred (documented in PARSER_DESIGN.md): lambda_symphony, calculator result
// multiplexing, block-lambda/ycombinator reduction, backpropagation statements.
// ============================================================================
#ifndef CARAMEL_PARSE_PARSER_H
#define CARAMEL_PARSE_PARSER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "caramel/ast/ast.h"
#include "caramel/parse/lexer.h"

namespace caramel::parse {

struct ParseError {
  std::string message;
  caramel::ast::SourceLocation loc;
};

class Parser {
 public:
  explicit Parser(std::vector<Token> tokens) : toks_(std::move(tokens)) {}

  // Parse a full compilation unit. Returns the Program (possibly partial on
  // error); diagnostics are in errors().
  std::unique_ptr<caramel::ast::Program> parse();

  const std::vector<ParseError>& errors() const { return errors_; }
  bool ok() const { return errors_.empty(); }

  // lang_044: non-fatal diagnostics (e.g. a plaintext `pass = "..."` in a
  // device:: block). Warnings never affect ok().
  const std::vector<ParseError>& warnings() const { return warnings_; }

 private:
  // --- token cursor ---
  const Token& peek(int ahead = 0) const;
  const Token& cur() const { return peek(0); }
  const Token& advance();
  bool check(TokKind k) const { return cur().kind == k; }
  bool match(TokKind k);
  bool at_end() const { return cur().kind == TokKind::EndOfFile; }
  void skip_newlines();
  void error(const std::string& msg);
  void warning(const std::string& msg);  // lang_044
  bool is_quant_directive_at_cursor() const;

  // lang_045: device aliases may contain hyphens (spec Section 2: "bob-i3").
  // The lexer splits "bob-i3" into Identifier "bob" + Identifier "-i3" (or
  // Number "-3" for digit tails), so alias positions greedily re-join
  // ADJACENT '-'-prefixed tokens (adjacency checked via source columns -
  // "bob - 3" never merges). `first` is the token that opened the name.
  void absorb_hyphen_suffix(const Token& first, std::string& name);

  // --- grammar productions ---
  std::unique_ptr<caramel::ast::Directive> parse_directive();
  // lang_044: device::alias { host = "..."; user = "..."; pass = env("N"); }
  std::unique_ptr<caramel::ast::DeviceBlock> parse_device_block();
  std::unique_ptr<caramel::ast::LambdaFlow> parse_lambda_flow();
  std::unique_ptr<caramel::ast::LambdaCalculator> parse_lambda_calculator();
  std::unique_ptr<caramel::ast::Block> parse_block();
  std::unique_ptr<caramel::ast::ClockSection> parse_clock_section();
  std::unique_ptr<caramel::ast::Stmt> parse_statement(std::vector<Token> line);
  std::vector<std::string> parse_param_list();

  // lang_006: consume leading @name(args)/@name{args} decorators at the cursor
  // (skipping newlines between them). @clock is left for block handling.
  std::vector<caramel::ast::Decorator> parse_decorators();

  // collect one logical statement's tokens up to a terminator at depth 0
  std::vector<Token> collect_statement_tokens();

  // --- RPN reduction over a value-token slice ---
  caramel::ast::ExprPtr reduce(const std::vector<Token>& value_tokens);

  // arity of an operator name: registry, then user table, else -1 (not an op)
  int arity_of(const std::string& name) const;

  std::vector<Token> toks_;
  std::size_t pos_ = 0;
  std::vector<ParseError> errors_;
  std::vector<ParseError> warnings_;  // lang_044
  std::unordered_map<std::string, int> user_fn_arity_;  // lambda_flow/symphony
  std::vector<caramel::ast::Decorator> pending_decorators_;  // lang_006
  bool seen_any_quant_ = false;
  bool seen_quantres_ = false;
  bool seen_quantmax_ = false;
  bool seen_quantmin_ = false;
  bool seen_quantrange_ = false;
  bool seen_legacy_quant_ = false;
};

}  // namespace caramel::parse

#endif  // CARAMEL_PARSE_PARSER_H
