// ============================================================================
// Caramel Language - CPU interpreter (simulation mode)
// ----------------------------------------------------------------------------
// Ticket:   lang_016 (Interpreter Architecture Design)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// A CPU reference VM that executes a Caramel program in software, for testing and
// validation without FPGA hardware. It runs the SSA dataflow graph (lang_014): a
// register file maps each SSA value id to a Value; nodes are evaluated in id order
// (operands always precede their uses in the graph). Parameters/free inputs are
// supplied by the caller; constants are read from the node lexeme.
//
// Architecture (this ticket): the VM, register file, control nodes, and a
// pluggable per-op evaluator with the scalar/elementwise arithmetic + relu
// built in. Full lambda evaluation is lang_017; matrix/tensor ops are lang_019;
// INT8 precision emulation is lang_018.
// ============================================================================
#ifndef CARAMEL_INTERP_INTERPRETER_H
#define CARAMEL_INTERP_INTERPRETER_H

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "caramel/interp/value.h"
#include "caramel/ir/dataflow.h"

namespace caramel::interp {

struct InterpError {
  std::string message;
};

// Result of running a program: output name -> Value, or an error.
struct RunResult {
  std::unordered_map<std::string, Value> outputs;
  std::optional<InterpError> error;
  bool ok() const { return !error.has_value(); }
};

// Pluggable evaluator for an operation node: given the op mnemonic and the
// already-evaluated operand values, produce the result (or nullopt if this op is
// not handled by this evaluator). lang_017/019 extend the op coverage by
// composing additional evaluators.
using OpEvaluator =
    std::function<std::optional<Value>(const std::string &op,
                                       const std::vector<Value> &operands,
                                       std::string *error)>;

// Param-aware evaluator: additionally receives the node's resolved integer op
// parameters (softmax {axis}, conv2d {stride}, ...). Tried before the plain
// evaluators so param-bearing ops (x86_057/058/059) can read them. Ops that
// take no param work equally through a plain OpEvaluator.
using ParamOpEvaluator = std::function<std::optional<Value>(
    const std::string &op, const std::vector<Value> &operands,
    const std::vector<std::pair<std::string, int64_t>> &params,
    std::string *error)>;

class Interpreter {
 public:
  Interpreter();

  // Provide a value for a flow parameter / free input (by name).
  void set_input(const std::string &name, Value v) { inputs_[name] = std::move(v); }

  // Install an additional op evaluator, tried before the built-ins (lang_017/019).
  void add_evaluator(OpEvaluator e) { evaluators_.push_back(std::move(e)); }

  // Install a param-aware evaluator, tried before the plain evaluators.
  void add_param_evaluator(ParamOpEvaluator e) {
    param_evaluators_.push_back(std::move(e));
  }

  // Optional post-op transform applied to every Op-node result, e.g. INT8
  // saturation to emulate the hardware datapath (lang_018).
  void set_op_result_transform(std::function<void(Value &)> f) {
    op_result_transform_ = std::move(f);
  }

  // Execute the dataflow graph; returns the named outputs (graph.outputs).
  RunResult run(const caramel::ir::DataflowGraph &graph);

 private:
  // Built-in scalar/elementwise arithmetic + relu (lang_016 baseline).
  static std::optional<Value> builtin_ops(const std::string &op,
                                           const std::vector<Value> &operands,
                                           std::string *error);

  std::unordered_map<std::string, Value> inputs_;
  std::vector<OpEvaluator> evaluators_;
  std::vector<ParamOpEvaluator> param_evaluators_;
  std::function<void(Value &)> op_result_transform_;
};

}  // namespace caramel::interp

#endif  // CARAMEL_INTERP_INTERPRETER_H
