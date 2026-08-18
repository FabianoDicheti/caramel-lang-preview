// ============================================================================
// Caramel Language - Abstract Syntax Tree (AST) node definitions
// ----------------------------------------------------------------------------
// Ticket:   lang_002 (AST Structure Definition)
// Version:  1.0.0
// Standard: C++17 (header-only declarations)
// ----------------------------------------------------------------------------
// This header defines the tree-shaped intermediate representation produced by
// the RPN parser (lang_005) from the surface syntax defined in
// grammar/caramel.ebnf (lang_001). It is the input to MLIR lowering (lang_009).
//
// Design notes (see knowledge/compiler/AST_DESIGN.md for the full rationale):
//   * Lambda-calculus core: the language is functions all the way down, so the
//     AST centers on application and abstraction.
//   * RPN reduction: the parser consumes the flat postfix term stream and uses
//     an operand stack to reduce operators with their operands into OpApp trees.
//     Stack operations (dup/swap/drop/...) are resolved during construction into
//     shared/duplicated subtrees; they do not survive as nodes in a fully
//     reduced AST, but a StackOp node is provided for unreduced/debug streams.
//   * Source locations are carried on every node for diagnostics.
//   * Object parameters survive as nodes here; positional resolution [A-7] is a
//     later pass (lang_004 owns operator signatures).
// ============================================================================
#ifndef CARAMEL_AST_AST_H
#define CARAMEL_AST_AST_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace caramel::ast {

// ---------------------------------------------------------------------------
// Source locations
// ---------------------------------------------------------------------------
struct SourceLocation {
  uint32_t line = 0;    // 1-based
  uint32_t column = 0;  // 1-based
  uint32_t offset = 0;  // 0-based byte offset into the source buffer
};

struct SourceRange {
  SourceLocation begin;
  SourceLocation end;
};

// ---------------------------------------------------------------------------
// Node kind discriminator (stable enum for fast visitation / serialization)
// ---------------------------------------------------------------------------
enum class NodeKind : uint16_t {
  // Expressions
  NumberLiteral,
  StringLiteral,
  TensorLiteral,
  VarRef,
  QualifiedRef,
  IndexedRef,
  PropertyRef,
  ObjectParams,
  LambdaExpr,
  OpApplication,
  StackOp,
  CombinatorRef,
  // Statements
  Assignment,
  PrintStatement,
  MemoryWrite,
  DeviceBlock,
  BackpropStatement,
  // Definitions and structure
  ClockSection,
  Block,
  LambdaFlow,
  LambdaSymphony,
  LambdaCalculator,
  Directive,
  Program,
};

// ---------------------------------------------------------------------------
// Base node
// ---------------------------------------------------------------------------
struct Node {
  explicit Node(NodeKind k) : kind(k) {}
  virtual ~Node() = default;

  NodeKind kind;
  SourceRange loc{};
};

using NodePtr = std::unique_ptr<Node>;

// Forward declarations
struct Expr;
struct Stmt;
using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

// ---------------------------------------------------------------------------
// Type annotations:  <namespace>::<name>{::<modifier>}   e.g. oc::tensor,
// ot::string::trunc  (grammar: type_annotation)
// ---------------------------------------------------------------------------
enum class TypeNamespace { Oc, Ot, Os };  // object-computable / tokenizable / storage

struct TypeAnnotation {
  TypeNamespace ns = TypeNamespace::Oc;
  std::string name;                      // number, tensor, string, text, dictionary, ...
  std::vector<std::string> modifiers;    // e.g. ["trunc"]
};

// ===========================================================================
// EXPRESSIONS
// ===========================================================================
struct Expr : Node {
  using Node::Node;
};

// Numeric literal. The raw lexeme is preserved; quantization to a fixed-point
// integer happens in a later pass using the program's quant directives.
struct NumberLiteral : Expr {
  NumberLiteral() : Expr(NodeKind::NumberLiteral) {}
  std::string lexeme;        // e.g. "-75.5"
  bool is_float = false;
};

enum class QuoteKind { Double, Single };

struct StringLiteral : Expr {
  StringLiteral() : Expr(NodeKind::StringLiteral) {}
  std::string value;
  QuoteKind quote = QuoteKind::Double;  // single quotes are used for file paths
};

// Nested square-bracket numeric list: scalar element or a sub-tensor.
struct TensorLiteral : Expr {
  TensorLiteral() : Expr(NodeKind::TensorLiteral) {}
  // Each element is either a NumberLiteral or a nested TensorLiteral.
  std::vector<ExprPtr> elements;
};

// Plain variable reference (an identifier that is NOT a reserved word, [A-9]).
struct VarRef : Expr {
  VarRef() : Expr(NodeKind::VarRef) {}
  std::string name;
};

// Namespaced reference: port::0, receiver::matrix_a, memory::W1, responser::out.
struct QualifiedRef : Expr {
  QualifiedRef() : Expr(NodeKind::QualifiedRef) {}
  std::string ns;       // port | receiver | responser | memory
  std::string member;   // identifier or integer-as-text (e.g. "0")
};

// One [ ... ] index: either a single integer or a [lo:hi] slice (either bound
// optional). is_slice distinguishes them.
struct Index {
  bool is_slice = false;
  std::optional<int64_t> first;   // index, or slice lower bound
  std::optional<int64_t> second;  // slice upper bound (slice only)
};

// C-style indexing/slicing on a base: A[0][1], x[0:32].
struct IndexedRef : Expr {
  IndexedRef() : Expr(NodeKind::IndexedRef) {}
  ExprPtr base;                 // VarRef or QualifiedRef
  std::vector<Index> indices;   // one per [ ... ]
};

// Property access producing a value: eigen_result.eigenvalues, matrix.properties
struct PropertyRef : Expr {
  PropertyRef() : Expr(NodeKind::PropertyRef) {}
  ExprPtr base;                     // typically a VarRef
  std::vector<std::string> path;    // ["eigenvalues"] or ["a","b"]
};

// Object parameters { key: value, ... }. Resolved to positional args later [A-7].
struct ObjectField {
  std::string key;
  ExprPtr value;
};

struct ObjectParams : Expr {
  ObjectParams() : Expr(NodeKind::ObjectParams) {}
  std::vector<ObjectField> fields;  // source order; semantics are order-independent
};

// Decorator (lang_006): @name(args) / @name{args} annotation attaching metadata
// or a transformation hint to a definition or statement. @clock is the canonical
// block-level decorator (modeled as ClockSection); other decorators attach to the
// node they precede. args is optional (a bare @name is allowed).
struct Decorator {
  std::string name;                       // e.g. "quantize", "tile", "device"
  std::unique_ptr<ObjectParams> args;     // null if the decorator took no args
  SourceRange loc{};
};

// Anonymous function. Two surface forms collapse to one node [A-8]:
//   arrow:  x => body         (form == Arrow, single param)
//   block:  lambda x => ... end  (form == Block; curried params nest as bodies)
enum class LambdaForm { Arrow, Block };

struct LambdaExpr : Expr {
  LambdaExpr() : Expr(NodeKind::LambdaExpr) {}
  LambdaForm form = LambdaForm::Arrow;
  std::vector<std::string> params;  // bound variables (curried order)
  ExprPtr body;                     // expression or nested LambdaExpr
};

// Operator / primitive application. After RPN reduction, args are the operands
// popped from the stack in left-to-right source order.
enum class OpClass { Arithmetic, Comparison, Logic, Primitive };

struct OpApplication : Expr {
  OpApplication() : Expr(NodeKind::OpApplication) {}
  std::string op;                  // canonical reserved name, e.g. "matmul"
  OpClass op_class = OpClass::Primitive;
  std::vector<ExprPtr> args;       // reduced operands (arity from lang_004 sig)
};

// Raw stack manipulation term. Present only in unreduced/debug streams; a fully
// reduced AST resolves these away (see header notes).
enum class StackOpKind { Dup, Swap, Drop, Over, Rot, RotReverse };

struct StackOp : Expr {
  StackOp() : Expr(NodeKind::StackOp) {}
  StackOpKind which = StackOpKind::Dup;
};

// Reference to a bird combinator used as an operand-producing term.
struct CombinatorRef : Expr {
  CombinatorRef() : Expr(NodeKind::CombinatorRef) {}
  std::string name;  // identity, kestrel, kite, bluebird, ycombinator, ...
};

// ===========================================================================
// STATEMENTS
// ===========================================================================
struct Stmt : Node {
  using Node::Node;
};

// RPN assignment: <value> <target(s)> [type] =
// targets holds one name, or several for tuple results [A-5].
struct Assignment : Stmt {
  Assignment() : Stmt(NodeKind::Assignment) {}
  std::vector<std::string> targets;
  std::optional<TypeAnnotation> type;
  ExprPtr value;  // reduced RHS expression
  std::vector<Decorator> decorators;  // lang_006: leading @decorators, if any
};

// "return( expr );" print/log form [A-3].
struct PrintStatement : Stmt {
  PrintStatement() : Stmt(NodeKind::PrintStatement) {}
  ExprPtr value;  // StringLiteral or any expression
};

// <value> memory::write   -- store a value back to a named memory slot.
struct MemoryWrite : Stmt {
  MemoryWrite() : Stmt(NodeKind::MemoryWrite) {}
  ExprPtr value;
  std::string slot;  // the memory::NAME being written (when statically known)
};

// crml::device::{fpga|cpu} { ... } -- explicit hardware-routing block; and
// (lang_044) device::alias { host = "..."; ... } -- a remote worker
// definition. Device::Remote marks the lang_044 form.
enum class Device { Fpga, Cpu, Remote };

// One `key = value;` entry of a device::alias block (lang_044). When
// `is_env` is set the source spelled env("NAME"): `value` then holds the
// environment-variable NAME, which is read at RUN time (never at parse
// time, and the resolved secret never lands in the AST).
struct DeviceField {
  std::string key;    // host | user | pass | token | port | kind | ...
  std::string value;  // string/number literal, or the env var name (is_env)
  bool is_env = false;
  SourceRange loc{};
};

struct DeviceBlock : Stmt {
  DeviceBlock() : Stmt(NodeKind::DeviceBlock) {}
  Device device = Device::Fpga;
  std::string alias;                 // device::alias definitions (lang_044)
  std::vector<DeviceField> fields;   // lang_044 `key = value;` entries
  std::vector<StmtPtr> body;         // crml::device:: routing form (legacy)
};

// <loss> crml::backpropagation { config } <targets> =
struct BackpropStatement : Stmt {
  BackpropStatement() : Stmt(NodeKind::BackpropStatement) {}
  ExprPtr loss;
  std::unique_ptr<ObjectParams> config;
  std::vector<std::string> targets;
};

// ===========================================================================
// STRUCTURE AND DEFINITIONS
// ===========================================================================
// A @clock(n): group of statements that run in parallel within one phase.
struct ClockSection : Node {
  ClockSection() : Node(NodeKind::ClockSection) {}
  int level = 0;
  std::vector<StmtPtr> statements;
};

// A brace-delimited body. It is either a flat list of statements or a sequence
// of clock sections; is_clocked selects which vector is meaningful.
struct Block : Node {
  Block() : Node(NodeKind::Block) {}
  bool is_clocked = false;
  std::vector<StmtPtr> statements;          // when !is_clocked
  std::vector<std::unique_ptr<ClockSection>> clocks;  // when is_clocked
};

struct LambdaFlow : Node {
  LambdaFlow() : Node(NodeKind::LambdaFlow) {}
  std::string name;
  std::vector<std::string> params;
  std::unique_ptr<Block> body;
  std::vector<std::string> returns;  // function-return clause name list [A-3]
  std::vector<Decorator> decorators;  // lang_006: leading @decorators, if any
};

struct LambdaSymphony : Node {
  LambdaSymphony() : Node(NodeKind::LambdaSymphony) {}
  std::string name;
  std::vector<std::string> params;
  std::unique_ptr<Block> body;
  std::vector<std::string> returns;
};

struct LambdaCalculator : Node {
  LambdaCalculator() : Node(NodeKind::LambdaCalculator) {}
  std::optional<int> channels;
  std::unique_ptr<Block> body;
};

// Compile-time configuration directive (crml::key = value; and friends [A-2]).
enum class DirectiveKind { Quant, Autodiff, Checkpoint, RoutingPolicy, RegisterBackward, Input, Status, Profile };

struct Directive : Node {
  Directive() : Node(NodeKind::Directive) {}
  DirectiveKind dkind = DirectiveKind::Quant;
  std::string key;                       // e.g. "quantmax"; for Input: the name
  std::string value;                     // raw value lexeme ("260", "true")
  std::unique_ptr<ObjectParams> object;  // for routing_policy
  std::vector<std::string> args;         // for register_backward(a, b)
  ExprPtr valueExpr;                     // for Input: `in::name = <literal>;`
};

// Root of a compilation unit.
struct Program : Node {
  Program() : Node(NodeKind::Program) {}
  std::vector<NodePtr> items;  // Directive | LambdaFlow | LambdaSymphony |
                               // LambdaCalculator | Stmt, in source order
};

// ===========================================================================
// VISITOR
// ===========================================================================
// Non-intrusive visitor. Concrete passes (type checker lang_003, MLIR lowering
// lang_009) derive from Visitor and override the nodes they care about. The
// default base dispatches on NodeKind; see AST_DESIGN.md for the traversal
// contract.
class Visitor {
 public:
  virtual ~Visitor() = default;

  // Expressions
  virtual void visit(NumberLiteral&) {}
  virtual void visit(StringLiteral&) {}
  virtual void visit(TensorLiteral&) {}
  virtual void visit(VarRef&) {}
  virtual void visit(QualifiedRef&) {}
  virtual void visit(IndexedRef&) {}
  virtual void visit(PropertyRef&) {}
  virtual void visit(ObjectParams&) {}
  virtual void visit(LambdaExpr&) {}
  virtual void visit(OpApplication&) {}
  virtual void visit(StackOp&) {}
  virtual void visit(CombinatorRef&) {}
  // Statements
  virtual void visit(Assignment&) {}
  virtual void visit(PrintStatement&) {}
  virtual void visit(MemoryWrite&) {}
  virtual void visit(DeviceBlock&) {}
  virtual void visit(BackpropStatement&) {}
  // Structure
  virtual void visit(ClockSection&) {}
  virtual void visit(Block&) {}
  virtual void visit(LambdaFlow&) {}
  virtual void visit(LambdaSymphony&) {}
  virtual void visit(LambdaCalculator&) {}
  virtual void visit(Directive&) {}
  virtual void visit(Program&) {}
};

// Dispatch a Node to the appropriate Visitor::visit overload by NodeKind.
void accept(Node& node, Visitor& visitor);

}  // namespace caramel::ast

#endif  // CARAMEL_AST_AST_H
