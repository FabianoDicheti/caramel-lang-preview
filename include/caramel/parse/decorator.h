// ============================================================================
// Caramel Language - Decorator system
// ----------------------------------------------------------------------------
// Ticket:   lang_006 (Decorator System)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Decorators are @name(args) / @name{args} annotations that attach metadata or a
// transformation hint to the definition or statement they precede. @clock is the
// canonical block-level decorator (handled as a ClockSection by the parser); the
// decorators here attach to flows and statements.
//
// This module defines the built-in decorator schemas and validation. The parser
// (lang_005) collects leading decorators and builds ast::Decorator nodes; the
// registry tells it which decorators exist, what they target, and which params
// they accept.
// ============================================================================
#ifndef CARAMEL_PARSE_DECORATOR_H
#define CARAMEL_PARSE_DECORATOR_H

#include <string>
#include <vector>

namespace caramel::parse {

// What a decorator may be attached to.
enum class DecoratorTarget {
  Statement,   // attaches to an assignment / op statement
  Definition,  // attaches to a lambda_flow / symphony / calculator
  Block,       // attaches to a brace block / clock section
  Any,
};

// Semantic family (used by later passes: quantization, scheduling, routing).
enum class DecoratorKind {
  Scheduling,    // clock
  Quantization,  // quantize
  Tiling,        // tile
  Routing,       // device
  Autodiff,      // autodiff
  Memory,        // prefetch
  User,          // user-defined / unknown
};

struct DecoratorSchema {
  std::string name;
  DecoratorTarget target;
  DecoratorKind kind;
  std::vector<std::string> param_keys;  // accepted object-param keys (advisory)
  bool args_required;                   // must it carry (args)/{args}?
};

class DecoratorRegistry {
 public:
  static const DecoratorRegistry& instance();

  // Schema for `name`, or nullptr if not a built-in decorator.
  const DecoratorSchema* lookup(const std::string& name) const;
  bool is_builtin(const std::string& name) const { return lookup(name) != nullptr; }

  std::size_t size() const { return table_.size(); }

 private:
  DecoratorRegistry();
  std::vector<DecoratorSchema> table_;
};

}  // namespace caramel::parse

#endif  // CARAMEL_PARSE_DECORATOR_H
