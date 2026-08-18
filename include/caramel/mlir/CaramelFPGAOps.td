//===- CaramelFPGAOps.td - Caramel FPGA dialect operations -*- tablegen -*-===//
//
// Ticket: lang_012 (Custom FPGA Dialect)
//
//===----------------------------------------------------------------------===//
#ifndef CARAMEL_FPGA_OPS
#define CARAMEL_FPGA_OPS

include "caramel/mlir/CaramelFPGADialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// caramel_fpga.matmul : INT8 systolic matrix multiply -> INT32 accumulator.
// Tiling attributes describe how the GEMM maps onto the 15x16 array; the
// verifier enforces the <= 32x32 tile bound from the IR contract.
def CaramelFPGA_MatMulOp : CaramelFPGA_Op<"matmul", [Pure]> {
  let summary = "INT8 weight-stationary systolic matmul (i8 x i8 -> i32)";
  let arguments = (ins
      AnyRankedTensor:$lhs,
      AnyRankedTensor:$rhs,
      DefaultValuedAttr<I64Attr, "15">:$tile_m,
      DefaultValuedAttr<I64Attr, "16">:$tile_n,
      DefaultValuedAttr<I64Attr, "16">:$tile_k);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let hasVerifier = 1;
}

// caramel_fpga.relu : activation (post-array), same shape/type in/out.
def CaramelFPGA_ReluOp : CaramelFPGA_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation max(x, 0)";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

// caramel_fpga.quantize : FP32 -> INT8 with scale/zero-point (host-side encode).
def CaramelFPGA_QuantizeOp : CaramelFPGA_Op<"quantize", [Pure]> {
  let summary = "Quantize FP32 -> INT8";
  let arguments = (ins AnyRankedTensor:$input,
                       F64Attr:$scale,
                       DefaultValuedAttr<I64Attr, "0">:$zero_point);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// caramel_fpga.dequantize : INT8/INT32 -> FP32 (host-side decode).
def CaramelFPGA_DequantizeOp : CaramelFPGA_Op<"dequantize", [Pure]> {
  let summary = "Dequantize INT -> FP32";
  let arguments = (ins AnyRankedTensor:$input,
                       F64Attr:$scale,
                       DefaultValuedAttr<I64Attr, "0">:$zero_point);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// caramel_fpga.sync : hardware barrier (barrier_id 0..15).
def CaramelFPGA_SyncOp : CaramelFPGA_Op<"sync"> {
  let summary = "Hardware synchronization barrier";
  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$barrier_id);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // CARAMEL_FPGA_OPS
