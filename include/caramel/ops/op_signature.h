// ============================================================================
// Caramel Language - Primitive operation catalog
// ----------------------------------------------------------------------------
// Ticket:   lang_004 (Primitive Operations)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// The authoritative table of built-in operations. Each OpSignature records the
// arity (how many operands the RPN reducer pops, [A-9]), the result count, a
// category, the object-parameter positional order ([A-7]), a default hardware
// routing target (from the language spec routing table), and a shape-inference
// hook (built on lang_003 primitives).
//
// Consumers:
//   * lang_005 parser - arity for OpApplication reduction
//   * lang_003/005     - result type inference
//   * lang_009 dialect - one MLIR op per primitive, routing as an attribute
// See knowledge/compiler/PRIMITIVE_OPS.md for the catalogue and semantics.
// ============================================================================
#ifndef CARAMEL_OPS_OP_SIGNATURE_H
#define CARAMEL_OPS_OP_SIGNATURE_H

#include <functional>
#include <string>
#include <vector>

#include "caramel/types/type.h"

namespace caramel::ops {

// Operation category (coarser than per-op; used for routing defaults and grouping).
enum class OpCategory {
  Arithmetic,     // add, sub, mul, div (elementwise scalars/tensors)
  Comparison,     // >, <, ==, >=, <=, !=
  Logic,          // and, or, not, xor
  Elementwise,    // elemwise_add/sub/mul/div, scalar_mul
  Activation,     // relu, sigmoid, tanh, gelu, silu, softmax, softplus
  Reduction,      // tensor_sum/mean/max/min, argmax
  Shape,          // transpose, reshape, concat, split
  LinAlg,         // matmul, determinant, inverse, eigendecomp, svd, qr, ...
  Convolution,    // conv1d/2d/3d, pooling
  Normalization,  // batchnorm, layernorm, instancenorm, groupnorm
  Attention,      // scaled_dot_attention, multihead_attention
  Sequence,       // selective_scan, s4_layer, fft, ifft, fft2d
  HostOp          // cross_entropy, exp, sqrt, embedding_lookup, get_timestamp
};

// Default hardware routing (overridable by crml::device blocks / routing_policy).
enum class HwTarget { Fpga, Cpu, Hybrid };

// Shape inference: given operand shapes (in source order), produce the result
// shape or a TypeError. Object-parameter values are not yet threaded in v1.0;
// ops whose result depends on params (conv/pool spatial dims) return a
// rank-correct dynamic shape and are refined downstream (documented per op).
using ShapeInferFn =
    std::function<caramel::types::Inferred<caramel::types::Shape>(
        const std::vector<caramel::types::Shape>&)>;

struct OpSignature {
  std::string name;
  int arity = 0;        // operands popped from the stack
  int results = 1;      // values produced
  OpCategory category = OpCategory::Elementwise;
  HwTarget target = HwTarget::Fpga;
  std::vector<std::string> param_keys;  // positional order for object params [A-7]
  ShapeInferFn infer_shape;             // may be null for control/host ops
};

// Global, immutable registry of built-in operations.
class OpRegistry {
 public:
  static const OpRegistry& instance();

  // Returns the signature for `name`, or nullptr if not a known primitive.
  const OpSignature* lookup(const std::string& name) const;

  // Convenience: arity of an op, or -1 if unknown.
  int arity_of(const std::string& name) const;

  std::size_t size() const { return table_.size(); }

 private:
  OpRegistry();
  std::vector<OpSignature> table_;
};

}  // namespace caramel::ops

#endif  // CARAMEL_OPS_OP_SIGNATURE_H
