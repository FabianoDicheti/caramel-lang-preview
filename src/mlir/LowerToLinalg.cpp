//===- LowerToLinalg.cpp - Caramel -> Linalg lowering --------------------===//
//
// Ticket: lang_010 (Lowering Pass: Caramel -> Linalg)
// Lowers the high-level caramel dialect (lang_009) to Linalg named ops on
// tensors, the next stage of the pipeline toward Affine (lang_011) and the FPGA
// dialect (lang_012):
//   caramel.matmul -> linalg.fill(0) + linalg.matmul
//   caramel.add    -> linalg.add
//   caramel.relu   -> linalg.max(x, zeros)   (relu = max(x, 0))
//
//===----------------------------------------------------------------------===//
#include "caramel/mlir/CaramelDialect.h"
#include "caramel/mlir/CaramelPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// An uninitialized result tensor of the given ranked type.
Value emptyLike(PatternRewriter &rw, Location loc, RankedTensorType t) {
  return rw.create<tensor::EmptyOp>(loc, t.getShape(), t.getElementType());
}

// A result tensor filled with the element-type zero (matmul accumulator / relu
// comparison operand).
Value zeroFilled(PatternRewriter &rw, Location loc, RankedTensorType t) {
  Value empty = emptyLike(rw, loc, t);
  Value zero = rw.create<arith::ConstantOp>(loc, rw.getZeroAttr(t.getElementType()));
  return rw.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty})
      .getResult(0);
}

struct MatMulLowering : OpRewritePattern<caramel::mlir::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(caramel::mlir::MatMulOp op,
                                PatternRewriter &rw) const override {
    auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resTy) return failure();
    Value init = zeroFilled(rw, op.getLoc(), resTy);
    auto mm = rw.create<linalg::MatmulOp>(
        op.getLoc(), TypeRange{resTy},
        ValueRange{op.getLhs(), op.getRhs()}, ValueRange{init});
    rw.replaceOp(op, mm.getResults());
    return success();
  }
};

struct AddLowering : OpRewritePattern<caramel::mlir::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(caramel::mlir::AddOp op,
                                PatternRewriter &rw) const override {
    auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resTy) return failure();
    Value init = emptyLike(rw, op.getLoc(), resTy);
    auto add = rw.create<linalg::AddOp>(
        op.getLoc(), TypeRange{resTy},
        ValueRange{op.getLhs(), op.getRhs()}, ValueRange{init});
    rw.replaceOp(op, add.getResults());
    return success();
  }
};

struct ReluLowering : OpRewritePattern<caramel::mlir::ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(caramel::mlir::ReluOp op,
                                PatternRewriter &rw) const override {
    auto resTy = dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!resTy) return failure();
    Value zeros = zeroFilled(rw, op.getLoc(), resTy);
    Value init = emptyLike(rw, op.getLoc(), resTy);
    auto mx = rw.create<linalg::MaxOp>(
        op.getLoc(), TypeRange{resTy},
        ValueRange{op.getInput(), zeros}, ValueRange{init});
    rw.replaceOp(op, mx.getResults());
    return success();
  }
};

struct LowerCaramelToLinalgPass
    : PassWrapper<LowerCaramelToLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCaramelToLinalgPass)

  StringRef getArgument() const final { return "lower-caramel-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower the caramel dialect to linalg named ops on tensors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulLowering, AddLowering, ReluLowering>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
    bool hasIllegalCaramelOp = false;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "caramel") {
        op->emitError("unsupported caramel op remains after lowering");
        hasIllegalCaramelOp = true;
      }
    });
    if (hasIllegalCaramelOp) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> caramel::mlir::createLowerCaramelToLinalgPass() {
  return std::make_unique<LowerCaramelToLinalgPass>();
}

// Explicit registration (called by caramel-opt) so the registration is not
// dropped when CaramelMLIR is linked as a static library.
void caramel::mlir::registerCaramelPasses() {
  ::mlir::PassRegistration<LowerCaramelToLinalgPass>();
}
