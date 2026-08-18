// ============================================================================
// Caramel Language - SSA dataflow graph construction
// ----------------------------------------------------------------------------
// Ticket:  lang_014 (SSA/Dataflow Model)
// Version: 1.0.0
// ============================================================================
#include "caramel/ir/dataflow.h"

#include <functional>

#include <algorithm>

namespace caramel::ir {

using namespace caramel::ast;

int DataflowGraph::numLevels() const {
  int n = 0;
  for (const auto &node : nodes) n = std::max(n, node.level + 1);
  return n;
}

std::vector<std::vector<ValueId>> DataflowGraph::levels() const {
  std::vector<std::vector<ValueId>> out(numLevels());
  for (const auto &node : nodes)
    if (node.level >= 0) out[node.level].push_back(node.id);
  return out;
}

namespace {

// Parse an object-param value expression to an int64 (NumberLiteral; float
// lexemes truncate toward zero). Returns false for non-numeric values.
bool paramInt(const Expr *e, int64_t &out) {
  if (!e || e->kind != ast::NodeKind::NumberLiteral) return false;
  const auto *n = static_cast<const NumberLiteral *>(e);
  try {
    out = static_cast<int64_t>(std::stoll(n->lexeme));
  } catch (...) {
    try { out = static_cast<int64_t>(std::stod(n->lexeme)); }
    catch (...) { return false; }
  }
  return true;
}

int64_t paramOr(const std::vector<std::pair<std::string, int64_t>> &ps,
                const std::string &key, int64_t dflt) {
  for (const auto &kv : ps) if (kv.first == key) return kv.second;
  return dflt;
}

// Fold resolved params into the opcode-defined 24-bit IR immediate
// (PROTOCOL_SPEC 6.5). The worker decodes with the mirror convention. Ops with
// no immediate param (or that carry params as tensor operands) return 0.
uint32_t paramImm(const std::string &op,
                  const std::vector<std::pair<std::string, int64_t>> &ps) {
  if (ps.empty()) return 0;
  // softmax: axis in [0..], 0 sentinel = "last axis"; imm = axis + 1.
  if (op == "softmax") {
    int64_t axis = paramOr(ps, "axis", -1);
    if (axis < 0) return 0;                 // default (last axis)
    return static_cast<uint32_t>((axis + 1) & 0x00FFFFFF);
  }
  // pooling: imm = kernel_size (low byte) | stride (next byte). stride defaults
  // to kernel_size (PyTorch default) when omitted.
  if (op == "maxpool2d" || op == "avgpool2d") {
    int64_t k = paramOr(ps, "kernel_size", 0);
    int64_t s = paramOr(ps, "stride", k);
    return static_cast<uint32_t>((k & 0xFF) | ((s & 0xFF) << 8));
  }
  // concat: axis, default 0; imm = axis + 1 (0 sentinel = axis 0).
  if (op == "concat") {
    int64_t axis = paramOr(ps, "axis", 0);
    return static_cast<uint32_t>((axis + 1) & 0x00FFFFFF);
  }
  // conv2d: imm = stride (low byte, default 1) | padding (next byte, default 0).
  if (op == "conv2d") {
    int64_t s = paramOr(ps, "stride", 1);
    int64_t p = paramOr(ps, "padding", 0);
    return static_cast<uint32_t>((s & 0xFF) | ((p & 0xFF) << 8));
  }
  return 0;
}

class Builder {
 public:
  explicit Builder(DataflowGraph &g) : g_(g) {}

  ValueId addNode(NodeKind kind, std::string op, std::vector<ValueId> operands) {
    DataflowNode n;
    n.id = static_cast<ValueId>(g_.nodes.size());
    n.kind = kind;
    n.op = std::move(op);
    n.operands = std::move(operands);
    // Dependency-depth level: one past the deepest operand (overridden by @clock).
    int lvl = 0;
    for (ValueId v : n.operands)
      if (v < g_.nodes.size()) lvl = std::max(lvl, g_.nodes[v].level + 1);
    n.level = lvl;
    g_.nodes.push_back(n);
    return n.id;
  }

  // Flatten a (possibly nested) tensor literal to row-major dims + data.
  // Returns false if the literal is ragged or holds a non-numeric element.
  static bool flattenTensorLiteral(const TensorLiteral &lit,
                                   std::vector<int64_t> &dims,
                                   std::vector<int64_t> &data) {
    dims.clear();
    data.clear();
    // Shape: walk the leading spine.
    const TensorLiteral *cur = &lit;
    while (true) {
      dims.push_back((int64_t)cur->elements.size());
      if (cur->elements.empty()) break;
      if (cur->elements[0]->kind != ast::NodeKind::TensorLiteral) break;
      cur = static_cast<const TensorLiteral *>(cur->elements[0].get());
    }
    // Data: depth-first, checking rectangularity against `dims` as we go.
    bool ok = true;
    std::function<void(const TensorLiteral &, size_t)> walk =
        [&](const TensorLiteral &t, size_t depth) {
          if (!ok) return;
          if (depth >= dims.size() || (int64_t)t.elements.size() != dims[depth]) {
            ok = false;
            return;
          }
          for (const auto &el : t.elements) {
            if (el->kind == ast::NodeKind::TensorLiteral) {
              walk(static_cast<const TensorLiteral &>(*el), depth + 1);
            } else if (el->kind == ast::NodeKind::NumberLiteral) {
              if (depth + 1 != dims.size()) { ok = false; return; }
              const auto *n = static_cast<const NumberLiteral *>(el.get());
              char *end = nullptr;
              long long v = std::strtoll(n->lexeme.c_str(), &end, 10);
              data.push_back((int64_t)v);
            } else {
              ok = false;
              return;
            }
          }
        };
    walk(lit, 0);
    return ok;
  }

  // Resolve an expression to the value id that produces it, creating nodes for
  // operations and constants as needed.
  ValueId lower(const Expr *e) {
    switch (e->kind) {
      case ast::NodeKind::NumberLiteral:
        return addNode(NodeKind::Const,
                       static_cast<const NumberLiteral *>(e)->lexeme, {});
      case ast::NodeKind::StringLiteral:
        return addNode(NodeKind::Const, "\"str\"", {});
      case ast::NodeKind::TensorLiteral: {
        // A tensor literal in a flow body is a real constant, not an external
        // input. Before this it fell through to the opaque-operand default and
        // became an unbound `<expr>`, which surfaced either as "missing input
        // value" or — worse — as a silently wrong result.
        const auto *lit = static_cast<const TensorLiteral *>(e);
        std::vector<int64_t> dims, data;
        if (!flattenTensorLiteral(*lit, dims, data))
          return addNode(NodeKind::Operand, "<malformed-tensor-literal>", {});
        ValueId id = addNode(NodeKind::Const, "<tensor>", {});
        g_.nodes[id].const_dims = std::move(dims);
        g_.nodes[id].const_data = std::move(data);
        return id;
      }
      case ast::NodeKind::VarRef: {
        const auto *v = static_cast<const VarRef *>(e);
        auto it = g_.bound.find(v->name);
        if (it != g_.bound.end()) return it->second;
        // free variable -> an external operand input
        ValueId id = addNode(NodeKind::Operand, v->name, {});
        g_.bound[v->name] = id;
        return id;
      }
      case ast::NodeKind::QualifiedRef: {
        const auto *q = static_cast<const QualifiedRef *>(e);
        std::string name = q->ns + "::" + q->member;
        auto it = g_.bound.find(name);
        if (it != g_.bound.end()) return it->second;
        ValueId id = addNode(NodeKind::Operand, name, {});
        g_.bound[name] = id;
        return id;
      }
      case ast::NodeKind::OpApplication: {
        const auto *o = static_cast<const OpApplication *>(e);
        std::vector<ValueId> operands;
        std::vector<std::pair<std::string, int64_t>> params;
        for (const auto &arg : o->args) {
          if (arg->kind == ast::NodeKind::ObjectParams) {
            // Resolve `{ key: value }` named params to integers (dropping
            // non-numeric ones); the interpreter reads these and the serializer
            // folds them into the IR immediate.
            const auto *op = static_cast<const ObjectParams *>(arg.get());
            for (const auto &f : op->fields) {
              int64_t iv;
              if (paramInt(f.value.get(), iv)) params.emplace_back(f.key, iv);
            }
            continue;
          }
          operands.push_back(lower(arg.get()));
        }
        ValueId id = addNode(NodeKind::Op, o->op, std::move(operands));
        g_.nodes[id].params = params;
        g_.nodes[id].imm = paramImm(o->op, params);
        return id;
      }
      default:
        // combinators / indexed / property refs: treat as opaque operand inputs
        return addNode(NodeKind::Operand, "<expr>", {});
    }
  }

  void lowerAssignment(const Assignment *a, int clockLevel) {
    if (!a->value) return;
    ValueId v = lower(a->value.get());
    if (clockLevel >= 0 && v < g_.nodes.size()) g_.nodes[v].level = clockLevel;
    // Bind every target name to the produced value (tuple targets share it in v1.0).
    for (const auto &t : a->targets) g_.bound[t] = v;
  }

 private:
  DataflowGraph &g_;
};

}  // namespace

DataflowGraph buildDataflow(const LambdaFlow &flow) {
  DataflowGraph g;
  Builder b(g);

  // Seed parameters as graph inputs (level 0).
  for (const auto &p : flow.params) {
    DataflowNode n;
    n.id = static_cast<ValueId>(g.nodes.size());
    n.kind = NodeKind::Param;
    n.op = p;
    n.level = 0;
    g.nodes.push_back(n);
    g.bound[p] = n.id;
  }

  if (flow.body) {
    if (flow.body->is_clocked) {
      for (const auto &cs : flow.body->clocks)
        for (const auto &stmt : cs->statements)
          if (stmt->kind == ast::NodeKind::Assignment)
            b.lowerAssignment(static_cast<const Assignment *>(stmt.get()),
                              cs->level);
    } else {
      for (const auto &stmt : flow.body->statements)
        if (stmt->kind == ast::NodeKind::Assignment)
          b.lowerAssignment(static_cast<const Assignment *>(stmt.get()), -1);
    }
  }

  g.outputs = flow.returns;
  return g;
}

}  // namespace caramel::ir
