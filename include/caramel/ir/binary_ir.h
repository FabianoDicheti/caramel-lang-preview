// ============================================================================
// Caramel Language - Binary IR serialization
// ----------------------------------------------------------------------------
// Ticket:   lang_015 (IR Serialization to Binary)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Serializes the SSA dataflow graph (lang_014) to the binary IR consumed by the
// caramel_os dataflow executor, following INTEGRATION_CONTRACT.md:
//   * 64-bit (8-byte) instructions
//   * 8-bit opcode
//   * up to 7 x 8-bit register/immediate fields
//
// Layout:
//   Header (16 bytes): magic "CRML", version u16, flags u16, instr_count u32,
//                      output_count u32
//   Instructions:      instr_count x 8 bytes
//   Output table:      output_count x 1 byte (value-id/register of each output)
//
// Instruction (8 bytes): opcode u8, dst u8, src0 u8, src1 u8, aux0 u8, aux1 u8,
//                        level u8, flags u8
// `level` carries the dataflow parallel level so the OS can issue a SYNC barrier
// between levels; `dst`/`src*` are SSA value ids used as register indices.
// ============================================================================
#ifndef CARAMEL_IR_BINARY_IR_H
#define CARAMEL_IR_BINARY_IR_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "caramel/ir/dataflow.h"

namespace caramel::ir {

// Binary opcodes (8-bit). Aligned with INTEGRATION_CONTRACT operation set; the
// numbering is the Caramel binary-IR ABI v1.
enum class OpCode : uint8_t {
  NOP = 0x00,
  PARAM = 0x01,    // graph input (param)
  CONST = 0x02,    // constant
  INPUT = 0x03,    // external operand input
  MATMUL = 0x10,
  ADD = 0x11,      // add / elemwise_add
  SUB = 0x12,
  MUL = 0x13,      // mul / elemwise_mul
  DIV = 0x14,
  SCALAR_MUL = 0x15,
  RELU = 0x20,
  SIGMOID = 0x21,
  TANH = 0x22,
  SOFTMAX = 0x23,
  CONV2D = 0x30,
  MAXPOOL2D = 0x31,
  AVGPOOL2D = 0x32,
  TRANSPOSE = 0x40,
  CONCAT = 0x41,
  RESHAPE = 0x42,  // reserved (needs a shape param tensor)
  SPLIT = 0x43,    // reserved (needs multi-output IR)
  QUANTIZE = 0x50,
  DEQUANTIZE = 0x51,
  LAYERNORM = 0x52,   // over the last axis (v1); affine gamma/beta reserved
  BATCHNORM = 0x53,   // reserved (5 operands > 4 registers)
  // 0x6x reduction class (whole-tensor -> rank-1 [1] scalar)
  TENSOR_SUM = 0x60,
  TENSOR_MEAN = 0x61,
  TENSOR_MAX = 0x62,
  TENSOR_MIN = 0x63,
  // 0x7x comparison class (elementwise -> 0/1)
  GT = 0x70,
  LT = 0x71,
  EQ = 0x72,
  GE = 0x73,
  LE = 0x74,
  NE = 0x75,
  // 0x8x logic class (elementwise -> 0/1; NOT is unary)
  AND = 0x80,
  OR = 0x81,
  XOR = 0x82,
  NOT = 0x83,
  SYNC = 0xF0,     // level barrier
  RETURN = 0xFE,
  UNKNOWN = 0xFF,
};

// Map a dataflow op mnemonic to a binary opcode (UNKNOWN if unmapped).
OpCode opcodeFor(const std::string &mnemonic);
const char *opcodeName(OpCode op);

struct Instruction {
  OpCode opcode = OpCode::NOP;
  uint8_t dst = 0;
  uint8_t src0 = 0;
  uint8_t src1 = 0;
  uint8_t aux0 = 0;
  uint8_t aux1 = 0;
  uint8_t level = 0;
  uint8_t flags = 0;
};

// IR v2 param channel (PROTOCOL_SPEC 6.5): the opcode-defined 24-bit immediate
// carried little-endian in aux0 (low) / aux1 (mid) / flags (high). Valid only
// on ops that use <= 2 operand registers; CONST is the precedent (it carries a
// 16-bit literal in aux0/aux1). Ops with no param leave all three bytes zero.
inline uint32_t instrImm24(const Instruction &ins) {
  return static_cast<uint32_t>(ins.aux0) |
         (static_cast<uint32_t>(ins.aux1) << 8) |
         (static_cast<uint32_t>(ins.flags) << 16);
}

struct Module {
  uint16_t version = 1;
  uint16_t flags = 0;
  std::vector<Instruction> instructions;
  std::vector<uint8_t> outputs;  // value-ids of graph outputs
};

// Serialize a dataflow graph to the binary IR byte stream.
std::vector<uint8_t> serialize(const DataflowGraph &graph);

// Parse a binary IR byte stream back into a Module (nullopt on malformed input).
// Used for round-trip validation and by tools/the OS-side reference reader.
std::optional<Module> deserialize(const std::vector<uint8_t> &bytes);

}  // namespace caramel::ir

#endif  // CARAMEL_IR_BINARY_IR_H
