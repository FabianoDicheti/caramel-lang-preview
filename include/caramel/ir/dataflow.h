// ============================================================================
// Caramel Language - SSA dataflow graph
// ----------------------------------------------------------------------------
// Ticket:   lang_014 (SSA/Dataflow Model)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// An SSA-based dataflow graph extracted from a lambda flow's AST. Each operation
// becomes a node producing one SSA value; edges are the def-use dependencies
// between nodes. Operations with no path between them at the same dependency
// depth are parallel — this is the information the caramel_os dataflow executor
// schedules on, and the structure lang_015 serializes to the binary IR.
//
// Built from the AST (lang_002), independent of the MLIR back end: the dataflow
// graph is the program's def-use graph and does not require the FPGA dialect.
// Explicit @clock(n) sections (lang_005) override the computed levels.
// ============================================================================
#ifndef CARAMEL_IR_DATAFLOW_H
#define CARAMEL_IR_DATAFLOW_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "caramel/ast/ast.h"

namespace caramel::ir {

using ValueId = uint32_t;
inline constexpr ValueId kInvalidValue = 0xFFFFFFFFu;

// The kind of a dataflow node.
enum class NodeKind {
  Param,    // a flow parameter (graph input)
  Const,    // a literal constant
  Op,       // an operation application (matmul, add, relu, ...)
  Operand,  // a free variable reference not otherwise defined (external input)
};

struct DataflowNode {
  ValueId id = kInvalidValue;     // the SSA value this node defines
  NodeKind kind = NodeKind::Op;
  std::string op;                 // op mnemonic (Op) or name/literal (Param/Const)
  std::vector<ValueId> operands;  // def-use edges: value ids consumed
  int level = -1;                 // parallel level (explicit @clock or computed)
  // IR v2 param channel (PROTOCOL_SPEC 6.5): an opcode-defined 24-bit immediate
  // carried in the instruction's aux0/aux1/flags bytes. Zero for ops that take
  // no param (the entire M1 set). Only usable when the op has <= 2 operands
  // (aux0/aux1 double as operand registers 3-4 otherwise); larger params ride
  // an extra INPUT "param tensor" operand instead. Derived from `params` at
  // build time (paramImm in dataflow.cpp) so the serializer just emits it.
  uint32_t imm = 0;
  // Resolved integer op parameters from a `{ key: value }` object-params term
  // (e.g. softmax {axis:0}, conv2d {stride:2}). The interpreter reads these
  // directly; the serializer folds them into `imm`. Empty for param-free ops.
  std::vector<std::pair<std::string, int64_t>> params;
  // Tensor constant payload, for a Const node lowered from a tensor literal
  // (`[2,3,5,7] from_spectrum`). Empty `const_dims` means the Const is a plain
  // scalar and its value is parsed from `op`, as it always was.
  std::vector<int64_t> const_dims;
  std::vector<int64_t> const_data;
};

struct DataflowGraph {
  std::vector<DataflowNode> nodes;                 // indexed by ValueId
  std::unordered_map<std::string, ValueId> bound;  // current name -> value id
  std::vector<std::string> outputs;                // returned names (graph outputs)

  // Number of distinct parallel levels (max level + 1).
  int numLevels() const;
  // Value ids grouped by level (each inner vector may run in parallel).
  std::vector<std::vector<ValueId>> levels() const;
};

// Build the dataflow graph for a lambda flow. Parameters seed the graph; each
// assignment's value expression is flattened into op nodes; the return clause
// records the graph outputs. If the flow body is @clock-partitioned, node levels
// follow the clock sections; otherwise they are computed by dependency depth.
DataflowGraph buildDataflow(const caramel::ast::LambdaFlow &flow);

}  // namespace caramel::ir

#endif  // CARAMEL_IR_DATAFLOW_H
