// ============================================================================
// Caramel Language - CPU interpreter implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_016 (Interpreter Architecture Design)
// Version: 1.0.0
// ============================================================================
#include "caramel/interp/interpreter.h"

#include "caramel/interp/generators.h"

#include <cstdlib>

namespace caramel::interp {

using caramel::ir::DataflowGraph;
using caramel::ir::DataflowNode;
using caramel::ir::NodeKind;
using caramel::ir::ValueId;

Interpreter::Interpreter() = default;

namespace {

// Apply a binary integer op elementwise, with scalar broadcasting.
std::optional<Value> elementwise(const Value &a, const Value &b,
                                 int64_t (*fn)(int64_t, int64_t),
                                 std::string *error) {
  if (a.is_scalar() && b.is_scalar())
    return Value::scalar(fn(a.data[0], b.data[0]));
  if (a.same_shape(b)) {
    Value r;
    r.dims = a.dims;
    r.data.resize(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) r.data[i] = fn(a.data[i], b.data[i]);
    return r;
  }
  // scalar broadcast against a tensor
  if (a.is_scalar() || b.is_scalar()) {
    const Value &t = a.is_scalar() ? b : a;
    const int64_t s = a.is_scalar() ? a.data[0] : b.data[0];
    Value r;
    r.dims = t.dims;
    r.data.resize(t.data.size());
    for (size_t i = 0; i < t.data.size(); ++i)
      r.data[i] = a.is_scalar() ? fn(s, t.data[i]) : fn(t.data[i], s);
    return r;
  }
  if (error) *error = "elementwise op operands have incompatible shapes";
  return std::nullopt;
}

}  // namespace

std::optional<Value> Interpreter::builtin_ops(const std::string &op,
                                              const std::vector<Value> &in,
                                              std::string *error) {
  // Tensor constructors (zeros/eye/diag/...). Same implementation as the
  // `in::m = 8 8 zeros;` initializer path, so a constructor means the same
  // thing wherever it appears.
  if (isGenerator(op)) return evalGenerator(op, in, error);

  auto need2 = [&]() {
    if (in.size() != 2) { if (error) *error = op + " expects 2 operands"; return false; }
    return true;
  };
  if (op == "add" || op == "elemwise_add") {
    if (!need2()) return std::nullopt;
    return elementwise(in[0], in[1], [](int64_t a, int64_t b) { return a + b; }, error);
  }
  if (op == "sub" || op == "elemwise_sub") {
    if (!need2()) return std::nullopt;
    return elementwise(in[0], in[1], [](int64_t a, int64_t b) { return a - b; }, error);
  }
  if (op == "mul" || op == "elemwise_mul") {
    if (!need2()) return std::nullopt;
    return elementwise(in[0], in[1], [](int64_t a, int64_t b) { return a * b; }, error);
  }
  if (op == "div" || op == "elemwise_div") {
    if (!need2()) return std::nullopt;
    return elementwise(in[0], in[1],
                       [](int64_t a, int64_t b) { return b != 0 ? a / b : 0; }, error);
  }
  if (op == "relu") {
    if (in.size() != 1) { if (error) *error = "relu expects 1 operand"; return std::nullopt; }
    Value r = in[0];
    for (auto &e : r.data) e = e < 0 ? 0 : e;
    return r;
  }
  return std::nullopt;  // not a built-in op (e.g. matmul -> lang_019)
}

RunResult Interpreter::run(const DataflowGraph &graph) {
  RunResult result;
  std::vector<std::optional<Value>> regs(graph.nodes.size());

  auto fail = [&](const std::string &m) {
    result.error = InterpError{m};
    return result;
  };

  for (const DataflowNode &node : graph.nodes) {
    switch (node.kind) {
      case NodeKind::Param:
      case NodeKind::Operand: {
        auto it = inputs_.find(node.op);
        if (it == inputs_.end())
          return fail("missing input value for '" + node.op + "'");
        regs[node.id] = it->second;
        break;
      }
      case NodeKind::Const: {
        // A tensor constant carries its payload on the node; a scalar constant
        // is parsed from the node lexeme. (Float literals are quantized
        // upstream; INT8 precision emulation is lang_018.)
        if (!node.const_dims.empty()) {
          regs[node.id] = Value::tensor(node.const_dims, node.const_data);
          break;
        }
        char *end = nullptr;
        long long v = std::strtoll(node.op.c_str(), &end, 10);
        regs[node.id] = Value::scalar(static_cast<int64_t>(v));
        break;
      }
      case NodeKind::Op: {
        std::vector<Value> operands;
        operands.reserve(node.operands.size());
        for (ValueId v : node.operands) {
          if (v >= regs.size() || !regs[v].has_value())
            return fail("operand value " + std::to_string(v) + " is not available");
          operands.push_back(*regs[v]);
        }
        std::string err;
        std::optional<Value> out;
        // Param-aware evaluators first (they see node.params); then plain ones.
        for (const auto &ev : param_evaluators_) {
          out = ev(node.op, operands, node.params, &err);
          if (out.has_value()) break;
        }
        if (!out.has_value()) {
          for (const auto &ev : evaluators_) {
            out = ev(node.op, operands, &err);
            if (out.has_value()) break;
          }
        }
        if (!out.has_value()) out = builtin_ops(node.op, operands, &err);
        if (!out.has_value())
          return fail(err.empty() ? "unsupported op '" + node.op + "'" : err);
        if (op_result_transform_) op_result_transform_(*out);  // lang_018 saturation
        regs[node.id] = std::move(out);
        break;
      }
    }
  }

  for (const std::string &name : graph.outputs) {
    auto it = graph.bound.find(name);
    if (it == graph.bound.end() || it->second >= regs.size() ||
        !regs[it->second].has_value())
      return fail("output '" + name + "' was not produced");
    result.outputs[name] = *regs[it->second];
  }
  return result;
}

}  // namespace caramel::interp
