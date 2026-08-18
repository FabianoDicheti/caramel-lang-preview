// ============================================================================
// Caramel Language - Primitive operation registry
// ----------------------------------------------------------------------------
// Ticket:  lang_004 (Primitive Operations)
// Version: 1.0.0
// ----------------------------------------------------------------------------
// Populates the built-in operation table. Shape-inference hooks reuse the
// lang_003 primitives (broadcast, infer_matmul, infer_elementwise).
// ============================================================================
#include "caramel/ops/op_signature.h"

#include <algorithm>

namespace caramel::ops {

using caramel::types::Inferred;
using caramel::types::Shape;

namespace {

// result == single operand's shape (activations, normalizations)
Inferred<Shape> same_shape(const std::vector<Shape>& in) {
  if (in.size() != 1) return Inferred<Shape>::bad("expected 1 operand");
  return Inferred<Shape>::good(in[0]);
}

// elementwise binary -> broadcast of the two operands
Inferred<Shape> elementwise2(const std::vector<Shape>& in) {
  if (in.size() != 2) return Inferred<Shape>::bad("expected 2 operands");
  return caramel::types::infer_elementwise(in[0], in[1]);
}

// matmul
Inferred<Shape> matmul2(const std::vector<Shape>& in) {
  if (in.size() != 2) return Inferred<Shape>::bad("expected 2 operands");
  return caramel::types::infer_matmul(in[0], in[1]);
}

// reduce-all -> scalar (rank 0)
Inferred<Shape> reduce_to_scalar(const std::vector<Shape>& in) {
  if (in.size() != 1) return Inferred<Shape>::bad("expected 1 operand");
  return Inferred<Shape>::good(Shape(std::vector<int64_t>{}));  // rank-0 scalar
}

Inferred<Shape> unary_dynamic(const std::vector<Shape>& in) {
  if (in.size() != 1) return Inferred<Shape>::bad("expected 1 operand");
  return Inferred<Shape>::good(Shape(std::vector<int64_t>(in[0].rank(), Shape::kDynamic)));
}

Inferred<Shape> binary_dynamic(const std::vector<Shape>& in) {
  if (in.size() != 2) return Inferred<Shape>::bad("expected 2 operands");
  return Inferred<Shape>::good(Shape(std::vector<int64_t>(in[0].rank(), Shape::kDynamic)));
}

Inferred<Shape> ternary_dynamic(const std::vector<Shape>& in) {
  if (in.size() != 3) return Inferred<Shape>::bad("expected 3 operands");
  return Inferred<Shape>::good(Shape(std::vector<int64_t>(in[0].rank(), Shape::kDynamic)));
}

// transpose -> swap last two dims (rank >= 2)
Inferred<Shape> transpose_last2(const std::vector<Shape>& in) {
  if (in.size() != 1) return Inferred<Shape>::bad("expected 1 operand");
  Shape s = in[0];
  if (s.rank() < 2) return Inferred<Shape>::bad("transpose requires rank >= 2");
  std::swap(s.dims[s.rank() - 1], s.dims[s.rank() - 2]);
  return Inferred<Shape>::good(s);
}

// Tensor constructors take scalar/vector operands whose VALUES (not shapes)
// determine the result, so a static shape cannot be inferred from operand
// shapes; report a rank-correct fully-dynamic shape. These are evaluated
// host-side at `in::name = <gen>` initialization time (caramel-run), so this
// hook only matters if a generator is (mis)used inside a flow body.
Inferred<Shape> ctor_matrix(const std::vector<Shape>&) {
  return Inferred<Shape>::good(Shape(std::vector<int64_t>{Shape::kDynamic, Shape::kDynamic}));
}
Inferred<Shape> ctor_vector(const std::vector<Shape>&) {
  return Inferred<Shape>::good(Shape(std::vector<int64_t>{Shape::kDynamic}));
}

// spatial ops whose exact result depends on object params (stride/padding):
// return a rank-correct, fully-dynamic shape; refined downstream once params
// are evaluated. Honest placeholder rather than a wrong concrete shape.
Inferred<Shape> dynamic_like(const std::vector<Shape>& in) {
  if (in.empty()) return Inferred<Shape>::bad("expected >= 1 operand");
  std::vector<int64_t> dims(in[0].rank(), Shape::kDynamic);
  return Inferred<Shape>::good(Shape(std::move(dims)));
}

}  // namespace

OpRegistry::OpRegistry() {
  auto add = [&](std::string name, int arity, OpCategory cat, HwTarget target,
                 std::vector<std::string> params, ShapeInferFn fn, int results = 1) {
    OpSignature sig;
    sig.name = std::move(name);
    sig.arity = arity;
    sig.results = results;
    sig.category = cat;
    sig.target = target;
    sig.param_keys = std::move(params);
    sig.infer_shape = std::move(fn);
    table_.push_back(std::move(sig));
  };

  // Arithmetic (scalar/tensor, broadcast)
  add("add", 2, OpCategory::Arithmetic, HwTarget::Fpga, {}, elementwise2);
  add("sub", 2, OpCategory::Arithmetic, HwTarget::Fpga, {}, elementwise2);
  add("mul", 2, OpCategory::Arithmetic, HwTarget::Fpga, {}, elementwise2);
  add("div", 2, OpCategory::Arithmetic, HwTarget::Fpga, {}, elementwise2);

  // Logic/comparison on broadcast-compatible operands.
  add("and", 2, OpCategory::Logic, HwTarget::Fpga, {}, elementwise2);
  add("or", 2, OpCategory::Logic, HwTarget::Fpga, {}, elementwise2);
  add("xor", 2, OpCategory::Logic, HwTarget::Fpga, {}, elementwise2);
  add("not", 1, OpCategory::Logic, HwTarget::Fpga, {}, same_shape);

  // Elementwise tensor ops
  add("elemwise_add", 2, OpCategory::Elementwise, HwTarget::Fpga, {}, elementwise2);
  add("elemwise_sub", 2, OpCategory::Elementwise, HwTarget::Fpga, {}, elementwise2);
  add("elemwise_mul", 2, OpCategory::Elementwise, HwTarget::Fpga, {}, elementwise2);
  add("elemwise_div", 2, OpCategory::Elementwise, HwTarget::Fpga, {}, elementwise2);
  add("scalar_mul", 2, OpCategory::Elementwise, HwTarget::Fpga, {}, elementwise2);

  // Activations (elementwise -> FPGA; transcendental softmax -> CPU per routing)
  add("relu", 1, OpCategory::Activation, HwTarget::Fpga, {}, same_shape);
  add("sigmoid", 1, OpCategory::Activation, HwTarget::Fpga, {}, same_shape);
  add("tanh", 1, OpCategory::Activation, HwTarget::Fpga, {}, same_shape);
  add("gelu", 1, OpCategory::Activation, HwTarget::Fpga, {}, same_shape);
  add("silu", 1, OpCategory::Activation, HwTarget::Fpga, {}, same_shape);
  add("softmax", 1, OpCategory::Activation, HwTarget::Cpu, {"axis"}, same_shape);
  add("softplus", 1, OpCategory::Activation, HwTarget::Cpu, {}, same_shape);

  // Linear algebra
  add("matmul", 2, OpCategory::LinAlg, HwTarget::Fpga, {}, matmul2);
  add("transpose", 1, OpCategory::Shape, HwTarget::Fpga, {}, transpose_last2);

  // Reductions -> scalar
  add("tensor_sum", 1, OpCategory::Reduction, HwTarget::Fpga, {}, reduce_to_scalar);
  add("tensor_mean", 1, OpCategory::Reduction, HwTarget::Fpga, {}, reduce_to_scalar);
  add("tensor_max", 1, OpCategory::Reduction, HwTarget::Fpga, {}, reduce_to_scalar);
  add("tensor_min", 1, OpCategory::Reduction, HwTarget::Fpga, {}, reduce_to_scalar);

  // Convolution / pooling (result shape depends on stride/padding params)
  add("conv1d", 2, OpCategory::Convolution, HwTarget::Fpga, {"stride", "padding"}, dynamic_like);
  add("conv2d", 2, OpCategory::Convolution, HwTarget::Fpga, {"stride", "padding"}, dynamic_like);
  add("conv3d", 2, OpCategory::Convolution, HwTarget::Fpga, {"stride", "padding"}, dynamic_like);
  add("maxpool2d", 1, OpCategory::Convolution, HwTarget::Fpga, {"kernel_size", "stride"}, dynamic_like);
  add("avgpool2d", 1, OpCategory::Convolution, HwTarget::Fpga, {"kernel_size", "stride"}, dynamic_like);
  add("adaptive_avgpool2d", 1, OpCategory::Convolution, HwTarget::Fpga, {"output_size"}, dynamic_like);

  // Normalization (precision-sensitive -> CPU/Hybrid per routing table)
  add("layernorm", 1, OpCategory::Normalization, HwTarget::Cpu, {"normalized_shape"}, same_shape);
  add("batchnorm", 5, OpCategory::Normalization, HwTarget::Hybrid, {"gamma", "beta"}, same_shape);
  add("instancenorm", 1, OpCategory::Normalization, HwTarget::Hybrid, {}, same_shape);
  add("groupnorm", 1, OpCategory::Normalization, HwTarget::Hybrid, {"num_groups"}, same_shape);

  // Shape and linalg ops not yet lowered, but reserved and parsed consistently.
  add("reshape", 1, OpCategory::Shape, HwTarget::Fpga, {"shape"}, unary_dynamic);
  add("concat", 2, OpCategory::Shape, HwTarget::Fpga, {"axis"}, binary_dynamic);
  add("split", 1, OpCategory::Shape, HwTarget::Fpga, {"axis", "sections"}, unary_dynamic, 2);
  add("argmax", 1, OpCategory::Reduction, HwTarget::Fpga, {"axis"}, reduce_to_scalar);
  add("determinant", 1, OpCategory::LinAlg, HwTarget::Cpu, {}, reduce_to_scalar);
  add("inverse", 1, OpCategory::LinAlg, HwTarget::Cpu, {}, same_shape);
  add("eigendecomp", 1, OpCategory::LinAlg, HwTarget::Cpu, {}, unary_dynamic, 2);
  add("svd", 1, OpCategory::LinAlg, HwTarget::Cpu, {}, unary_dynamic, 3);
  add("qr", 1, OpCategory::LinAlg, HwTarget::Cpu, {}, unary_dynamic, 2);
  add("cholesky", 1, OpCategory::LinAlg, HwTarget::Cpu, {}, same_shape);
  add("lu", 1, OpCategory::LinAlg, HwTarget::Cpu, {}, unary_dynamic, 3);
  add("fft", 1, OpCategory::Sequence, HwTarget::Cpu, {}, same_shape);
  add("ifft", 1, OpCategory::Sequence, HwTarget::Cpu, {}, same_shape);
  add("fft2d", 1, OpCategory::Sequence, HwTarget::Cpu, {}, same_shape);
  add("scaled_dot_attention", 3, OpCategory::Attention, HwTarget::Hybrid, {}, ternary_dynamic);
  add("multihead_attention", 3, OpCategory::Attention, HwTarget::Hybrid,
      {"num_heads", "embed_dim"}, ternary_dynamic);
  add("selective_scan", 2, OpCategory::Sequence, HwTarget::Hybrid, {}, binary_dynamic);
  add("s4_layer", 2, OpCategory::Sequence, HwTarget::Hybrid, {}, binary_dynamic);

  // Host-side ops
  add("cross_entropy", 2, OpCategory::HostOp, HwTarget::Cpu, {}, reduce_to_scalar);
  add("clip_gradient", 1, OpCategory::HostOp, HwTarget::Cpu, {"max_norm"}, same_shape);
  add("exp", 1, OpCategory::HostOp, HwTarget::Cpu, {}, same_shape);
  add("sqrt", 1, OpCategory::HostOp, HwTarget::Cpu, {}, same_shape);
  add("embedding_lookup", 2, OpCategory::HostOp, HwTarget::Cpu, {}, binary_dynamic);
  add("tensor.init", 0, OpCategory::HostOp, HwTarget::Cpu, {"shape", "dtype"}, nullptr);
  add("txt.read", 1, OpCategory::HostOp, HwTarget::Cpu, {}, same_shape);
  add("get_timestamp", 0, OpCategory::HostOp, HwTarget::Cpu, {}, nullptr);

  // Tensor constructors (lang: host-side generators). RPN postfix form, e.g.
  // `in::m = 8 8 zeros;`, `in::v = 100 range;`, `in::d = [2,3,5] diag;`.
  // Evaluated at initialization time by caramel-run; see literal_to_value /
  // eval_initializer. Arities are fixed (the RPN reducer needs them).
  add("zeros", 2, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);   // rows cols
  add("ones", 2, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);    // rows cols
  add("full", 3, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);    // rows cols value
  add("eye", 1, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);     // n
  add("diag", 1, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);    // [vector]
  add("range", 1, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_vector);   // n
  add("random", 2, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);  // rows cols
  add("band", 4, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);    // n lower upper fill
  add("tridiag", 4, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix); // n sub diag super
  add("from_spectrum", 1, OpCategory::HostOp, HwTarget::Cpu, {}, ctor_matrix);  // [eigenvalues]
}

const OpRegistry& OpRegistry::instance() {
  static const OpRegistry registry;
  return registry;
}

const OpSignature* OpRegistry::lookup(const std::string& name) const {
  auto it = std::find_if(table_.begin(), table_.end(),
                         [&](const OpSignature& s) { return s.name == name; });
  return it == table_.end() ? nullptr : &*it;
}

int OpRegistry::arity_of(const std::string& name) const {
  const OpSignature* s = lookup(name);
  return s ? s->arity : -1;
}

}  // namespace caramel::ops
