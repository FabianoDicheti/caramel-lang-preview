// RUN: caramel-opt %s | caramel-opt | FileCheck %s
// Ticket: lang_009 (Caramel MLIR Dialect)
// Round-trips the Caramel dialect ops through caramel-opt (parse -> verify ->
// print -> reparse) and checks the printed form with FileCheck.

// CHECK-LABEL: func.func @matmul_relu_add
func.func @matmul_relu_add(%a: tensor<2x3xi32>, %b: tensor<3x4xi32>,
                           %bias: tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: caramel.matmul %{{.*}}, %{{.*}} : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
  %0 = caramel.matmul %a, %b : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
  // CHECK: caramel.add %{{.*}}, %{{.*}} : tensor<2x4xi32>
  %1 = caramel.add %0, %bias : tensor<2x4xi32>
  // CHECK: caramel.relu %{{.*}} : tensor<2x4xi32>
  %2 = caramel.relu %1 : tensor<2x4xi32>
  return %2 : tensor<2x4xi32>
}
