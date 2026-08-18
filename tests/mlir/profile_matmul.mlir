// RUN: caramel-opt --lower-caramel-to-fpga --profile-matmul %s | FileCheck %s
// Ticket: lang_013 (Matrix Profiling Pass)
// Profiles matmuls for the 15x16 array: MAC count, tile count, estimated cycles,
// and flags the hotspot (highest-MAC matmul).

// CHECK-LABEL: func.func @two_matmuls
func.func @two_matmuls(%a: tensor<30x40xi8>, %b: tensor<40x48xi8>,
                       %c: tensor<2x3xi8>, %d: tensor<3x4xi8>)
    -> (tensor<30x48xi32>, tensor<2x4xi32>) {
  // Big matmul: 30*48*40 = 57600 MACs; tiles = ceil(30/15)*ceil(48/16)*ceil(40/16)
  //           = 2*3*3 = 18; cycles = 18*(15+16+16) = 846. It is the hotspot.
  // CHECK: caramel_fpga.matmul
  // CHECK-SAME: profile.cycles_est = 846
  // CHECK-SAME: profile.hotspot
  // CHECK-SAME: profile.macs = 57600
  // CHECK-SAME: profile.tiles = 18
  %0 = caramel.matmul %a, %b : (tensor<30x40xi8>, tensor<40x48xi8>) -> tensor<30x48xi32>
  // Small matmul: 2*4*3 = 24 MACs; not the hotspot (no profile.hotspot attr).
  // CHECK: caramel_fpga.matmul
  // CHECK-SAME: profile.macs = 24
  // CHECK-NOT: profile.hotspot
  %1 = caramel.matmul %c, %d : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi32>
  return %0, %1 : tensor<30x48xi32>, tensor<2x4xi32>
}
