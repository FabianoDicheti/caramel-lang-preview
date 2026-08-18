// RUN: caramel-opt --lower-caramel-to-linalg %s | FileCheck %s
// Ticket: lang_010 (Lowering Pass: Caramel -> Linalg)
// Checks that the caramel dialect lowers to Linalg named ops on tensors.

// CHECK-LABEL: func.func @lower
func.func @lower(%a: tensor<2x3xi32>, %b: tensor<3x4xi32>,
                 %bias: tensor<2x4xi32>) -> tensor<2x4xi32> {
  // matmul -> fill(0) + linalg.matmul
  // CHECK: linalg.fill
  // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<2x3xi32>, tensor<3x4xi32>) outs(%{{.*}} : tensor<2x4xi32>)
  %0 = caramel.matmul %a, %b : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
  // add -> linalg.add
  // CHECK: linalg.add ins(%{{.*}}, %{{.*}} : tensor<2x4xi32>, tensor<2x4xi32>)
  %1 = caramel.add %0, %bias : tensor<2x4xi32>
  // relu -> fill(0) + linalg.max
  // CHECK: linalg.max ins(%{{.*}}, %{{.*}} : tensor<2x4xi32>, tensor<2x4xi32>)
  %2 = caramel.relu %1 : tensor<2x4xi32>
  // CHECK-NOT: caramel.
  return %2 : tensor<2x4xi32>
}
