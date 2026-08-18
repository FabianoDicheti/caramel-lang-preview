//===- CaramelFPGADialect.cpp - Caramel FPGA dialect impl ----------------===//
//
// Ticket: lang_012 (Custom FPGA Dialect)
//
//===----------------------------------------------------------------------===//
#include "caramel/mlir/CaramelFPGADialect.h"

#include <algorithm>

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace caramel_fpga;

#include "caramel/mlir/CaramelFPGAOpsDialect.cpp.inc"

void CaramelFPGADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "caramel/mlir/CaramelFPGAOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "caramel/mlir/CaramelFPGAOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifiers - encode the hardware constraints (15x16 INT8 array, tile <= 32).
//===----------------------------------------------------------------------===//
LogicalResult MatMulOp::verify() {
  auto lhs = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhs = llvm::cast<RankedTensorType>(getRhs().getType());
  auto res = llvm::cast<RankedTensorType>(getResult().getType());

  if (lhs.getRank() < 2 || rhs.getRank() < 2)
    return emitOpError("operands must have rank >= 2");
  if (res.getRank() < 2)
    return emitOpError("result must have rank >= 2");

  // INT8 inputs accumulate into INT32 (the PE MAC datapath).
  if (!lhs.getElementType().isInteger(8) || !rhs.getElementType().isInteger(8))
    return emitOpError("operands must be INT8 (i8) tensors for the systolic array");
  if (!res.getElementType().isInteger(32))
    return emitOpError("result must be an INT32 (i32) accumulator tensor");

  const int64_t kLhs = lhs.getShape().back();
  const int64_t kRhs = rhs.getShape()[rhs.getRank() - 2];
  if (!ShapedType::isDynamic(kLhs) && !ShapedType::isDynamic(kRhs) &&
      kLhs != kRhs)
    return emitOpError("inner dimensions disagree: ") << kLhs << " vs " << kRhs;

  const int64_t m = lhs.getShape()[lhs.getRank() - 2];
  const int64_t n = rhs.getShape().back();
  const int64_t resultM = res.getShape()[res.getRank() - 2];
  const int64_t resultN = res.getShape().back();
  if (!ShapedType::isDynamic(m) && !ShapedType::isDynamic(resultM) &&
      m != resultM)
    return emitOpError("result M dimension mismatch");
  if (!ShapedType::isDynamic(n) && !ShapedType::isDynamic(resultN) &&
      n != resultN)
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
    int64_t lhsDim = li >= 0 ? lhsBatch[li] : 1;
    int64_t rhsDim = ri >= 0 ? rhsBatch[ri] : 1;
    int64_t expected = ShapedType::kDynamic;
    if (ShapedType::isDynamic(lhsDim) || ShapedType::isDynamic(rhsDim)) {
      expected = ShapedType::kDynamic;
    } else if (lhsDim == rhsDim) {
      expected = lhsDim;
    } else if (lhsDim == 1) {
      expected = rhsDim;
    } else if (rhsDim == 1) {
      expected = lhsDim;
    } else {
      return emitOpError("batch dimensions are not broadcast-compatible");
    }
    if (!ShapedType::isDynamic(expected) && !ShapedType::isDynamic(resBatch[i]) &&
        expected != resBatch[i])
      return emitOpError("result batch dimension mismatch");
  }

  // Tile bounds from the IR contract (CONTEXT_FROM_LANG: safe tiles <= 32x32).
  if (getTileM() <= 0 || getTileM() > kMaxTile)
    return emitOpError("tile_m must be in 1..") << kMaxTile;
  if (getTileN() <= 0 || getTileN() > kMaxTile)
    return emitOpError("tile_n must be in 1..") << kMaxTile;
  if (getTileK() <= 0 || getTileK() > kMaxTile)
    return emitOpError("tile_k must be in 1..") << kMaxTile;
  return success();
}

LogicalResult SyncOp::verify() {
  // Hardware barrier ids are 0..15 (CONTEXT_FROM_LANG).
  if (getBarrierId() < 0 || getBarrierId() > 15)
    return emitOpError("barrier_id must be in 0..15");
  return success();
}
