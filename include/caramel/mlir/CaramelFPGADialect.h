//===- CaramelFPGADialect.h - Caramel FPGA dialect C++ interface ---------===//
//
// Ticket: lang_012 (Custom FPGA Dialect)
//
//===----------------------------------------------------------------------===//
#ifndef CARAMEL_MLIR_CARAMELFPGADIALECT_H
#define CARAMEL_MLIR_CARAMELFPGADIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "caramel/mlir/CaramelFPGAOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "caramel/mlir/CaramelFPGAOps.h.inc"

namespace caramel_fpga {
// Hardware parameters of the caramel_board accelerator (board_015).
constexpr int64_t kArrayRows = 15;     // PE rows
constexpr int64_t kArrayCols = 16;     // PE cols (240 PEs total)
constexpr int64_t kMaxTile = 32;       // IR-contract tile bound (<= 32x32)

// Register a convenience pipeline / passes for the FPGA dialect (lang_012).
void registerCaramelFPGAPasses();

// Register the matrix profiling pass (lang_013).
void registerCaramelProfilePasses();
}  // namespace caramel_fpga

#endif // CARAMEL_MLIR_CARAMELFPGADIALECT_H
