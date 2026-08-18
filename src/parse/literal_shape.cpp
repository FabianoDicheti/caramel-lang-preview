// ============================================================================
// Caramel Language - Matrix/tensor literal shape inference
// ----------------------------------------------------------------------------
// Ticket:  lang_007 (Matrix Initialization Syntax)
// Version: 1.0.0
// ============================================================================
#include "caramel/parse/literal_shape.h"

namespace caramel::parse {

using caramel::ast::NodeKind;
using caramel::ast::TensorLiteral;
using caramel::types::Shape;

namespace {

// Recursively infer the dimension vector. Returns false on a ragged / mixed
// literal and writes a message into `error`.
bool inferDims(const TensorLiteral &lit, std::vector<int64_t> &dims,
               std::string *error) {
  const std::size_t n = lit.elements.size();
  dims.push_back(static_cast<int64_t>(n));
  if (n == 0) return true;  // empty level: shape ends here

  // All elements must be the same kind: all scalars (leaf) or all sub-tensors.
  const bool firstIsTensor = lit.elements[0]->kind == NodeKind::TensorLiteral;
  for (std::size_t i = 0; i < n; ++i) {
    const bool isTensor = lit.elements[i]->kind == NodeKind::TensorLiteral;
    if (isTensor != firstIsTensor) {
      if (error)
        *error = "ragged tensor literal: element " + std::to_string(i) +
                 " mixes scalars and sub-tensors";
      return false;
    }
  }

  if (!firstIsTensor) return true;  // leaf row of scalars

  // All sub-tensors must share the same shape.
  std::vector<int64_t> subDims;
  if (!inferDims(static_cast<const TensorLiteral &>(*lit.elements[0]), subDims,
                 error))
    return false;
  for (std::size_t i = 1; i < n; ++i) {
    std::vector<int64_t> otherDims;
    if (!inferDims(static_cast<const TensorLiteral &>(*lit.elements[i]),
                   otherDims, error))
      return false;
    if (otherDims != subDims) {
      if (error)
        *error = "ragged tensor literal: row " + std::to_string(i) +
                 " has a different shape than row 0";
      return false;
    }
  }
  dims.insert(dims.end(), subDims.begin(), subDims.end());
  return true;
}

}  // namespace

std::optional<Shape> inferTensorLiteralShape(const TensorLiteral &lit,
                                             std::string *error) {
  std::vector<int64_t> dims;
  if (!inferDims(lit, dims, error)) return std::nullopt;
  return Shape(std::move(dims));
}

bool isRectangular(const TensorLiteral &lit) {
  return inferTensorLiteralShape(lit, nullptr).has_value();
}

}  // namespace caramel::parse
