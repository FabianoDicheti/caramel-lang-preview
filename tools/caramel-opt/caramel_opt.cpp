//===- caramel_opt.cpp - The caramel-opt driver --------------------------===//
//
// Ticket: lang_009 (Caramel MLIR Dialect)
// An mlir-opt-style driver that registers the Caramel dialect (plus the upstream
// dialects) so .mlir files using `caramel.*` ops can be parsed, verified, and
// round-tripped. The lowering passes (lang_010+) register here too.
//
//===----------------------------------------------------------------------===//
#include "caramel/mlir/CaramelDialect.h"
#include "caramel/mlir/CaramelFPGADialect.h"
#include "caramel/mlir/CaramelPasses.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// lang_011: convenience pipeline lowering the caramel dialect all the way to
// affine loops (caramel -> linalg -> one-shot-bufferize -> linalg-to-affine).
static mlir::PassPipelineRegistration<> kCaramelToAffine(
    "caramel-to-affine",
    "Lower the caramel dialect to affine loops "
    "(caramel -> linalg -> bufferize -> affine)",
    [](mlir::OpPassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(
          caramel::mlir::createLowerCaramelToLinalgPass());
      mlir::bufferization::OneShotBufferizationOptions opts;
      opts.bufferizeFunctionBoundaries = true;
      pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::createConvertLinalgToAffineLoopsPass());
    });

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<caramel::mlir::CaramelDialect>();
  registry.insert<caramel_fpga::CaramelFPGADialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  caramel::mlir::registerCaramelPasses();
  caramel_fpga::registerCaramelFPGAPasses();
  caramel_fpga::registerCaramelProfilePasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "caramel-opt: Caramel MLIR driver\n", registry));
}
