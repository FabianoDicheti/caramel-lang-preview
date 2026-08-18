//===- CaramelDialect.cpp - Caramel dialect implementation ---------------===//
//
// Ticket: lang_009 (Caramel MLIR Dialect)
//
//===----------------------------------------------------------------------===//
#include "caramel/mlir/CaramelDialect.h"

#include <algorithm>

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace caramel::mlir;

// Dialect definitions (generated).
#include "caramel/mlir/CaramelOpsDialect.cpp.inc"

void CaramelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "caramel/mlir/CaramelOps.cpp.inc"
      >();
}

// Op definitions (generated).
#define GET_OP_CLASSES
#include "caramel/mlir/CaramelOps.cpp.inc"

//===----------------------------------------------------------------------===//
// MatMulOp verifier
//===----------------------------------------------------------------------===//
LogicalResult MatMulOp::verify() {
  auto lhs = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhs = llvm::cast<RankedTensorType>(getRhs().getType());
  auto res = llvm::cast<RankedTensorType>(getResult().getType());
  if (lhs.getRank() < 2 || rhs.getRank() < 2)
    return emitOpError("operands must have rank >= 2");
  if (res.getRank() < 2)
    return emitOpError("result must have rank >= 2");
  const int64_t k_lhs = lhs.getShape().back();
  const int64_t k_rhs = rhs.getShape()[rhs.getRank() - 2];
  if (!ShapedType::isDynamic(k_lhs) && !ShapedType::isDynamic(k_rhs) &&
      k_lhs != k_rhs)
    return emitOpError("inner dimensions disagree: ")
           << k_lhs << " vs " << k_rhs;
  const int64_t m = lhs.getShape()[lhs.getRank() - 2];
  const int64_t n = rhs.getShape().back();
  const int64_t rm = res.getShape()[res.getRank() - 2];
  const int64_t rn = res.getShape().back();
  if (!ShapedType::isDynamic(m) && !ShapedType::isDynamic(rm) && m != rm)
    return emitOpError("result M dimension mismatch");
  if (!ShapedType::isDynamic(n) && !ShapedType::isDynamic(rn) && n != rn)
    return emitOpError("result N dimension mismatch");
  auto lhsBatch = lhs.getShape().drop_back(2);
  auto rhsBatch = rhs.getShape().drop_back(2);
  auto resBatch = res.getShape().drop_back(2);
  if (resBatch.size() != std::max(lhsBatch.size(), rhsBatch.size()))
    return emitOpError("result batch rank mismatch");
  for (int64_t i = static_cast<int64_t>(resBatch.size()) - 1,
               li = static_cast<int64_t>(lhsBatch.size()) - 1,
               ri = static_cast<int64_t>(rhsBatch.size()) - 1;
       i >= 0; --i, --li, --ri) {
    int64_t ld = li >= 0 ? lhsBatch[li] : 1;
    int64_t rd = ri >= 0 ? rhsBatch[ri] : 1;
    int64_t expected = ShapedType::kDynamic;
    if (ShapedType::isDynamic(ld) || ShapedType::isDynamic(rd)) {
      expected = ShapedType::kDynamic;
    } else if (ld == rd) {
      expected = ld;
    } else if (ld == 1) {
      expected = rd;
    } else if (rd == 1) {
      expected = ld;
    } else {
      return emitOpError("batch dimensions are not broadcast-compatible");
    }
    if (!ShapedType::isDynamic(expected) && !ShapedType::isDynamic(resBatch[i]) &&
        expected != resBatch[i])
      return emitOpError("result batch dimension mismatch");
  }
  return success();
}
