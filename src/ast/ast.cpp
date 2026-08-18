// ============================================================================
// Caramel Language - AST visitor dispatch
// ----------------------------------------------------------------------------
// Ticket:  lang_002 (AST Structure Definition)
// Version: 1.0.0
// ----------------------------------------------------------------------------
// Implements accept(): a single, centralized dispatch from a base Node to the
// correct Visitor overload, keyed by NodeKind. Keeping dispatch here (rather
// than a virtual accept() on every node) keeps the node structs plain data and
// makes serialization (lang_015) straightforward.
// ============================================================================
#include "caramel/ast/ast.h"

namespace caramel::ast {

void accept(Node& node, Visitor& visitor) {
  switch (node.kind) {
    // Expressions
    case NodeKind::NumberLiteral:
      return visitor.visit(static_cast<NumberLiteral&>(node));
    case NodeKind::StringLiteral:
      return visitor.visit(static_cast<StringLiteral&>(node));
    case NodeKind::TensorLiteral:
      return visitor.visit(static_cast<TensorLiteral&>(node));
    case NodeKind::VarRef:
      return visitor.visit(static_cast<VarRef&>(node));
    case NodeKind::QualifiedRef:
      return visitor.visit(static_cast<QualifiedRef&>(node));
    case NodeKind::IndexedRef:
      return visitor.visit(static_cast<IndexedRef&>(node));
    case NodeKind::PropertyRef:
      return visitor.visit(static_cast<PropertyRef&>(node));
    case NodeKind::ObjectParams:
      return visitor.visit(static_cast<ObjectParams&>(node));
    case NodeKind::LambdaExpr:
      return visitor.visit(static_cast<LambdaExpr&>(node));
    case NodeKind::OpApplication:
      return visitor.visit(static_cast<OpApplication&>(node));
    case NodeKind::StackOp:
      return visitor.visit(static_cast<StackOp&>(node));
    case NodeKind::CombinatorRef:
      return visitor.visit(static_cast<CombinatorRef&>(node));
    // Statements
    case NodeKind::Assignment:
      return visitor.visit(static_cast<Assignment&>(node));
    case NodeKind::PrintStatement:
      return visitor.visit(static_cast<PrintStatement&>(node));
    case NodeKind::MemoryWrite:
      return visitor.visit(static_cast<MemoryWrite&>(node));
    case NodeKind::DeviceBlock:
      return visitor.visit(static_cast<DeviceBlock&>(node));
    case NodeKind::BackpropStatement:
      return visitor.visit(static_cast<BackpropStatement&>(node));
    // Structure and definitions
    case NodeKind::ClockSection:
      return visitor.visit(static_cast<ClockSection&>(node));
    case NodeKind::Block:
      return visitor.visit(static_cast<Block&>(node));
    case NodeKind::LambdaFlow:
      return visitor.visit(static_cast<LambdaFlow&>(node));
    case NodeKind::LambdaSymphony:
      return visitor.visit(static_cast<LambdaSymphony&>(node));
    case NodeKind::LambdaCalculator:
      return visitor.visit(static_cast<LambdaCalculator&>(node));
    case NodeKind::Directive:
      return visitor.visit(static_cast<Directive&>(node));
    case NodeKind::Program:
      return visitor.visit(static_cast<Program&>(node));
  }
}

}  // namespace caramel::ast
