// RUN: not caramel-opt %s 2>&1 | FileCheck %s
// Ticket: lang_012 (Custom FPGA Dialect)
// Verifies direct caramel_fpga IR cannot bypass GEMM shape checks.

func.func @bad_inner_dim(%a: tensor<2x3xi8>, %b: tensor<4x5xi8>) -> tensor<2x5xi32> {
  // CHECK: error: 'caramel_fpga.matmul' op inner dimensions disagree: 3 vs 4
  %0 = caramel_fpga.matmul %a, %b : (tensor<2x3xi8>, tensor<4x5xi8>) -> tensor<2x5xi32>
  return %0 : tensor<2x5xi32>
}
