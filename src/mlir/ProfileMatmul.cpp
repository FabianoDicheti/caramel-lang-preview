//===- ProfileMatmul.cpp - Matrix profiling pass -------------------------===//
//
// Ticket: lang_013 (Matrix Profiling Pass)
// Analyzes matmul operations (both the high-level `caramel.matmul` and the
// hardware `caramel_fpga.matmul`) and attaches cost metadata used to guide tiling
// and hardware mapping onto the 15x16 INT8 systolic array:
//   profile.macs       - multiply-accumulate count (M*N*K)
//   profile.tiles      - array tiles needed: ceil(M/15)*ceil(N/16)*ceil(K/tile_k)
//   profile.cycles_est - estimated array cycles (simple fill + accumulate model)
//   profile.hotspot    - unit attr on the highest-MAC matmul in the function
//
//===----------------------------------------------------------------------===//
#include "caramel/mlir/CaramelDialect.h"
#include "caramel/mlir/CaramelFPGADialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct MatmulCost {
  Operation *op = nullptr;
  int64_t macs = 0;
  int64_t tiles = 0;
  int64_t cycles = 0;
};

int64_t ceilDiv(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : a; }

// Extract [M, N, K] from a matmul's lhs/result shapes; returns false if dynamic.
bool gemmDims(Value lhs, Value result, int64_t &m, int64_t &n, int64_t &k) {
  auto lt = dyn_cast<RankedTensorType>(lhs.getType());
  auto rt = dyn_cast<RankedTensorType>(result.getType());
  if (!lt || !rt || lt.getRank() < 2 || rt.getRank() < 2) return false;
  if (!lt.hasStaticShape() || !rt.hasStaticShape()) return false;
  m = rt.getShape()[rt.getRank() - 2];
  n = rt.getShape().back();
  k = lt.getShape().back();
  return true;
}

MatmulCost profile(Operation *op, int64_t m, int64_t n, int64_t k, int64_t tk) {
  MatmulCost c;
  c.op = op;
  c.macs = m * n * k;
  // Tiles to cover the GEMM on the 15x16 array.
  c.tiles = ceilDiv(m, caramel_fpga::kArrayRows) *
            ceilDiv(n, caramel_fpga::kArrayCols) * ceilDiv(k, tk > 0 ? tk : k);
  // Simple cycle model: each tile pays a pipeline fill (rows+cols) plus tk
  // accumulation cycles.
  const int64_t fill = caramel_fpga::kArrayRows + caramel_fpga::kArrayCols;
  c.cycles = c.tiles * (fill + (tk > 0 ? tk : k));
  return c;
}

struct ProfileMatmulPass
    : PassWrapper<ProfileMatmulPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProfileMatmulPass)

  StringRef getArgument() const final { return "profile-matmul"; }
  StringRef getDescription() const final {
    return "Profile matmul ops (MAC count, array tiles, estimated cycles) and "
           "flag the hotspot, to guide tiling/hardware mapping";
  }

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    OpBuilder b(&getContext());
    std::vector<MatmulCost> costs;

    fn.walk([&](Operation *op) {
      int64_t m, n, k;
      if (auto fm = dyn_cast<caramel_fpga::MatMulOp>(op)) {
        if (gemmDims(fm.getLhs(), fm.getResult(), m, n, k))
          costs.push_back(profile(op, m, n, k, fm.getTileK()));
      } else if (auto cm = dyn_cast<caramel::mlir::MatMulOp>(op)) {
        if (gemmDims(cm.getLhs(), cm.getResult(), m, n, k))
          costs.push_back(profile(op, m, n, k, /*tk=*/caramel_fpga::kArrayCols));
      }
    });

    int64_t maxMacs = -1;
    Operation *hot = nullptr;
    for (auto &c : costs) {
      c.op->setAttr("profile.macs", b.getI64IntegerAttr(c.macs));
      c.op->setAttr("profile.tiles", b.getI64IntegerAttr(c.tiles));
      c.op->setAttr("profile.cycles_est", b.getI64IntegerAttr(c.cycles));
      if (c.macs > maxMacs) { maxMacs = c.macs; hot = c.op; }
    }
    if (hot) hot->setAttr("profile.hotspot", b.getUnitAttr());
  }
};

}  // namespace

namespace caramel_fpga {
void registerCaramelProfilePasses() {
  ::mlir::PassRegistration<ProfileMatmulPass>();
}
}  // namespace caramel_fpga
