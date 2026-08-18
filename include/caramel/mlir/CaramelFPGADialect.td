//===- CaramelFPGADialect.td - Caramel FPGA dialect --------*- tablegen -*-===//
//
// Ticket: lang_012 (Custom FPGA Dialect)
// Hardware-ISA dialect modeling the caramel_board accelerator: a 15x16 (240-PE)
// weight-stationary INT8 systolic array on Kintex-7 (see
// caramel_board/knowledge/hardware/ARCHITECTURE_RATIONALE.md and board_015).
// Ops mirror the board IR opcode set (CONTEXT_FROM_LANG.md): MATMUL, RELU,
// QUANTIZE, DEQUANTIZE, TILE, SYNC. This is the lowering target between the
// high-level `caramel` dialect (lang_009) and the binary IR (lang_015).
//
//===----------------------------------------------------------------------===//
#ifndef CARAMEL_FPGA_DIALECT
#define CARAMEL_FPGA_DIALECT

include "mlir/IR/OpBase.td"

def CaramelFPGA_Dialect : Dialect {
  let name = "caramel_fpga";
  let summary = "Hardware-ISA dialect for the Caramel 15x16 INT8 systolic array";
  let description = [{
    Operations that map 1:1 to the caramel_board accelerator ISA. INT8 inputs
    accumulate into INT32 (the PE MAC width). The systolic array is 15 rows x 16
    columns (240 PEs = 240 Kintex-7 DSPs); GEMMs larger than a tile are split by
    the TILE op, with the IR contract bounding tiles to <= 32x32.
  }];
  let cppNamespace = "::caramel_fpga";
}

class CaramelFPGA_Op<string mnemonic, list<Trait> traits = []>
    : Op<CaramelFPGA_Dialect, mnemonic, traits>;

#endif // CARAMEL_FPGA_DIALECT
