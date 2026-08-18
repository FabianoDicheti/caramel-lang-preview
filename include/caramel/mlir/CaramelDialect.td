//===- CaramelDialect.td - Caramel dialect definition ------*- tablegen -*-===//
//
// Ticket: lang_009 (Caramel MLIR Dialect)
// The high-level Caramel dialect: the first MLIR lowering target for programs
// produced by the front end (lang_005). Lowered to Linalg in lang_010.
//
//===----------------------------------------------------------------------===//
#ifndef CARAMEL_DIALECT
#define CARAMEL_DIALECT

include "mlir/IR/OpBase.td"

def Caramel_Dialect : Dialect {
  let name = "caramel";
  let summary = "High-level Caramel tensor/ML dialect";
  let description = [{
    Operations mirroring the Caramel primitive set (lang_004): integer-quantized
    tensor algebra and activations. Ops carry a hardware-routing attribute so the
    routing pass (lang_013) can target FPGA vs CPU.
  }];
  let cppNamespace = "::caramel::mlir";
}

// Base class for all Caramel ops.
class Caramel_Op<string mnemonic, list<Trait> traits = []>
    : Op<Caramel_Dialect, mnemonic, traits>;

#endif // CARAMEL_DIALECT
