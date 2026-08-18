//===- LowerToFPGA.cpp - caramel -> caramel_fpga lowering ----------------===//
//
// Ticket: lang_012 (Custom FPGA Dialect)
// Lowers the high-level `caramel` dialect (lang_009) to the hardware-ISA
// `caramel_fpga` dialect for the 15x16 INT8 systolic array. Ops that have a
// hardware mapping (matmul, relu) lower; ops without one (e.g. elementwise add)
// are left for host/CPU routing and not rewritten here.
//
//===----------------------------------------------------------------------===//
#include "caramel/mlir/CaramelDialect.h"
#include "caramel/mlir/CaramelFPGADialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// caramel.matmul (i8 x i8 -> i32) -> caramel_fpga.matmul with array tiling attrs.
struct MatMulToFPGA : OpRewritePattern<caramel::mlir::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(caramel::mlir::MatMulOp op,
                                PatternRewriter &rw) const override {
    auto lhs = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhs = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto res = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!lhs || !rhs || !res) return failure();
    // Only INT8->INT32 matmuls map to the systolic array; others route to host.
    if (!lhs.getElementType().isInteger(8) ||
        !rhs.getElementType().isInteger(8) ||
        !res.getElementType().isInteger(32))
      return failure();
    rw.replaceOpWithNewOp<caramel_fpga::MatMulOp>(
        op, res, op.getLhs(), op.getRhs(),
        /*tile_m=*/caramel_fpga::kArrayRows,
        /*tile_n=*/caramel_fpga::kArrayCols,
        /*tile_k=*/caramel_fpga::kArrayCols);
    return success();
  }
};

// caramel.relu -> caramel_fpga.relu (same type).
struct ReluToFPGA : OpRewritePattern<caramel::mlir::ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(caramel::mlir::ReluOp op,
                                PatternRewriter &rw) const override {
    rw.replaceOpWithNewOp<caramel_fpga::ReluOp>(op, op.getResult().getType(),
                                                 op.getInput());
    return success();
  }
};

struct LowerCaramelToFPGAPass
    : PassWrapper<LowerCaramelToFPGAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCaramelToFPGAPass)

  StringRef getArgument() const final { return "lower-caramel-to-fpga"; }
  StringRef getDescription() const final {
    return "Lower the caramel dialect to the caramel_fpga hardware dialect "
           "(15x16 INT8 systolic array)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<caramel_fpga::CaramelFPGADialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulToFPGA, ReluToFPGA>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void caramel_fpga::registerCaramelFPGAPasses() {
  ::mlir::PassRegistration<LowerCaramelToFPGAPass>();
}
