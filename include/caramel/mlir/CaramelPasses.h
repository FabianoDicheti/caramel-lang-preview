//===- CaramelPasses.h - Caramel dialect passes --------------------------===//
//
// Ticket: lang_010 (Lowering Pass: Caramel -> Linalg)
//
//===----------------------------------------------------------------------===//
#ifndef CARAMEL_MLIR_CARAMELPASSES_H
#define CARAMEL_MLIR_CARAMELPASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace caramel::mlir {

// Lowers caramel.{matmul,add,relu} to the Linalg dialect (on tensors).
std::unique_ptr<::mlir::Pass> createLowerCaramelToLinalgPass();

// Registers all Caramel passes with the global pass registry (call this from a
// driver such as caramel-opt so they appear as command-line options). Provided as
// an explicit call so the registration object is not dropped from the static lib.
void registerCaramelPasses();

}  // namespace caramel::mlir

#endif  // CARAMEL_MLIR_CARAMELPASSES_H
