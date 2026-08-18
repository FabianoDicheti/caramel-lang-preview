// RUN: caramel-opt --caramel-to-affine %s | FileCheck %s
// Ticket: lang_011 (Lowering Pass: Linalg -> Affine)
// The caramel-to-affine pipeline lowers the caramel dialect through linalg and
// bufferization to explicit affine loop nests (hardware-mappable form).

// CHECK-LABEL: func.func @mm
func.func @mm(%a: tensor<2x3xi32>, %b: tensor<3x4xi32>) -> tensor<2x4xi32> {
  %0 = caramel.matmul %a, %b : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}
// matmul becomes a triple-nested affine loop with a multiply-accumulate body.
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.for
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store
// CHECK-NOT: caramel.
// CHECK-NOT: linalg.
