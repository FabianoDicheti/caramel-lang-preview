//===- CaramelOps.td - Caramel dialect operations ----------*- tablegen -*-===//
//
// Ticket: lang_009 (Caramel MLIR Dialect)
//
//===----------------------------------------------------------------------===//
#ifndef CARAMEL_OPS
#define CARAMEL_OPS

include "caramel/mlir/CaramelDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// caramel.matmul : [.., M, K] x [.., K, N] -> [.., M, N]
def Caramel_MatMulOp : Caramel_Op<"matmul", [Pure]> {
  let summary = "Integer-quantized matrix multiply";
  let description = [{
    Maps the front-end `matmul` primitive. Operands and result are ranked tensors;
    shape compatibility (inner dim agreement) is checked by the verifier.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let hasVerifier = 1;
}

// caramel.add : elementwise add, same type in/out
def Caramel_AddOp : Caramel_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise add";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// caramel.relu : ReLU activation, same type in/out
def Caramel_ReluOp : Caramel_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation max(x, 0)";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

#endif // CARAMEL_OPS
