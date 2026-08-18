// RUN: caramel-opt --lower-caramel-to-fpga %s | FileCheck %s
// Ticket: lang_012 (Custom FPGA Dialect)
// Lowers the high-level caramel dialect to the caramel_fpga hardware dialect for
// the 15x16 INT8 systolic array. INT8 matmul + relu map to the array; the matmul
// carries the array tiling attributes.

// CHECK-LABEL: func.func @f
func.func @f(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi32> {
  // INT8 matmul lowers to the systolic-array op (tile attrs default to the 15x16
  // array dims and are elided by the printer when equal to their defaults).
  // CHECK: caramel_fpga.matmul %{{.*}}, %{{.*}} : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi32>
  %0 = caramel.matmul %a, %b : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi32>
  // CHECK: caramel_fpga.relu %{{.*}} : tensor<2x4xi32>
  %1 = caramel.relu %0 : tensor<2x4xi32>
  // CHECK-NOT: caramel.matmul
  // CHECK-NOT: caramel.relu
  return %1 : tensor<2x4xi32>
}
