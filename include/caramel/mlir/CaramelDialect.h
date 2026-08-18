//===- CaramelDialect.h - Caramel dialect C++ interface -------------------===//
//
// Ticket: lang_009 (Caramel MLIR Dialect)
//
//===----------------------------------------------------------------------===//
#ifndef CARAMEL_MLIR_CARAMELDIALECT_H
#define CARAMEL_MLIR_CARAMELDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect declaration (generated from CaramelDialect.td).
#include "caramel/mlir/CaramelOpsDialect.h.inc"

// Op declarations (generated from CaramelOps.td).
#define GET_OP_CLASSES
#include "caramel/mlir/CaramelOps.h.inc"

#endif // CARAMEL_MLIR_CARAMELDIALECT_H
