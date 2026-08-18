// ============================================================================
// Caramel Language - MLIR environment smoke tool
// ----------------------------------------------------------------------------
// Ticket:  lang_008 (MLIR Environment Setup)
// ----------------------------------------------------------------------------
// Minimal program that exercises the MLIR build integration end to end: create
// an MLIRContext, programmatically build a `builtin.module` containing an empty
// `func.func @caramel_probe()`, verify it, and print it. If this links, runs,
// and verifies, the MLIR toolchain is correctly wired for the dialect work in
// lang_009.
// ============================================================================
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // builtin.module { func.func @caramel_probe() { return } }
  auto module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  auto funcType = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto func = builder.create<mlir::func::FuncOp>(loc, "caramel_probe", funcType);
  auto& entry = *func.addEntryBlock();
  builder.setInsertionPointToEnd(&entry);
  builder.create<mlir::func::ReturnOp>(loc);

  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "caramel-mlir-smoke: module verification FAILED\n";
    return 1;
  }

  module.print(llvm::outs());
  llvm::outs() << "\ncaramel-mlir-smoke: MLIR environment OK ("
               << "func.func built and verified)\n";
  return 0;
}
