// ============================================================================
// Caramel Language - Binary IR serialization
// ----------------------------------------------------------------------------
// Ticket:  lang_015 (IR Serialization to Binary)
// Version: 1.0.0
// ============================================================================
#include "caramel/ir/binary_ir.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace caramel::ir {

namespace {
constexpr std::array<char, 4> kMagic = {'C', 'R', 'M', 'L'};
constexpr std::size_t kHeaderSize = 16;
constexpr std::size_t kInstrSize = 8;

void put_u16(std::vector<uint8_t> &b, uint16_t v) {
  b.push_back(static_cast<uint8_t>(v & 0xFF));
  b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
void put_u32(std::vector<uint8_t> &b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
uint16_t get_u16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
uint32_t get_u32(const uint8_t *p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
  return v;
}

// v1.0 uses 8-bit register fields. Reject graphs that cannot be represented
// instead of silently aliasing distinct SSA values.
uint8_t reg(ValueId v) {
  if (v > std::numeric_limits<uint8_t>::max()) {
    throw std::overflow_error("binary IR v1 supports at most 256 SSA values");
  }
  return static_cast<uint8_t>(v);
}

uint8_t level(int n) {
  if (n < 0) return 0;
  if (n > std::numeric_limits<uint8_t>::max()) {
    throw std::overflow_error("binary IR v1 supports dataflow levels 0..255");
  }
  return static_cast<uint8_t>(n);
}

ValueId outputValue(const DataflowGraph &graph, const std::string &name) {
  auto it = graph.bound.find(name);
  if (it == graph.bound.end()) {
    throw std::invalid_argument("dataflow output is not bound: " + name);
  }
  return it->second;
}

void validateGraph(const DataflowGraph &graph) {
  if (graph.nodes.size() > static_cast<std::size_t>(std::numeric_limits<uint8_t>::max()) + 1) {
    throw std::overflow_error("binary IR v1 supports at most 256 SSA values");
  }
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto &n = graph.nodes[i];
    if (n.id != i) {
      throw std::invalid_argument("dataflow node id does not match node index");
    }
    (void)level(n.level);
    if (n.operands.size() > 4) {
      throw std::invalid_argument("binary IR v1 encodes at most four operands per instruction");
    }
    if (n.kind == NodeKind::Op && opcodeFor(n.op) == OpCode::UNKNOWN) {
      throw std::invalid_argument("operation has no binary IR opcode: " + n.op);
    }
    for (ValueId operand : n.operands) {
      if (operand >= graph.nodes.size()) {
        throw std::invalid_argument("dataflow operand references an undefined SSA value");
      }
      (void)reg(operand);
    }
  }
  for (const auto &name : graph.outputs) (void)reg(outputValue(graph, name));
}
}  // namespace

OpCode opcodeFor(const std::string &m) {
  if (m == "matmul") return OpCode::MATMUL;
  if (m == "add" || m == "elemwise_add") return OpCode::ADD;
  if (m == "sub" || m == "elemwise_sub") return OpCode::SUB;
  if (m == "mul" || m == "elemwise_mul") return OpCode::MUL;
  if (m == "div" || m == "elemwise_div") return OpCode::DIV;
  if (m == "scalar_mul") return OpCode::SCALAR_MUL;
  if (m == "tensor_sum") return OpCode::TENSOR_SUM;
  if (m == "tensor_mean") return OpCode::TENSOR_MEAN;
  if (m == "tensor_max") return OpCode::TENSOR_MAX;
  if (m == "tensor_min") return OpCode::TENSOR_MIN;
  // Comparisons: the frontend carries operator symbols; accept the worker
  // mnemonics too so either form serializes to the same opcode.
  if (m == ">"  || m == "gt") return OpCode::GT;
  if (m == "<"  || m == "lt") return OpCode::LT;
  if (m == "==" || m == "eq") return OpCode::EQ;
  if (m == ">=" || m == "ge") return OpCode::GE;
  if (m == "<=" || m == "le") return OpCode::LE;
  if (m == "!=" || m == "ne") return OpCode::NE;
  if (m == "and") return OpCode::AND;
  if (m == "or") return OpCode::OR;
  if (m == "xor") return OpCode::XOR;
  if (m == "not") return OpCode::NOT;
  if (m == "relu") return OpCode::RELU;
  if (m == "sigmoid") return OpCode::SIGMOID;
  if (m == "tanh") return OpCode::TANH;
  if (m == "softmax") return OpCode::SOFTMAX;
  if (m == "conv2d") return OpCode::CONV2D;
  if (m == "maxpool2d") return OpCode::MAXPOOL2D;
  if (m == "avgpool2d") return OpCode::AVGPOOL2D;
  if (m == "transpose") return OpCode::TRANSPOSE;
  if (m == "concat") return OpCode::CONCAT;
  if (m == "reshape") return OpCode::RESHAPE;
  if (m == "split") return OpCode::SPLIT;
  if (m == "quantize") return OpCode::QUANTIZE;
  if (m == "dequantize") return OpCode::DEQUANTIZE;
  if (m == "layernorm") return OpCode::LAYERNORM;
  if (m == "batchnorm") return OpCode::BATCHNORM;
  return OpCode::UNKNOWN;
}

const char *opcodeName(OpCode op) {
  switch (op) {
    case OpCode::NOP: return "NOP";
    case OpCode::PARAM: return "PARAM";
    case OpCode::CONST: return "CONST";
    case OpCode::INPUT: return "INPUT";
    case OpCode::MATMUL: return "MATMUL";
    case OpCode::ADD: return "ADD";
    case OpCode::SUB: return "SUB";
    case OpCode::MUL: return "MUL";
    case OpCode::DIV: return "DIV";
    case OpCode::SCALAR_MUL: return "SCALAR_MUL";
    case OpCode::RELU: return "RELU";
    case OpCode::SIGMOID: return "SIGMOID";
    case OpCode::TANH: return "TANH";
    case OpCode::SOFTMAX: return "SOFTMAX";
    case OpCode::CONV2D: return "CONV2D";
    case OpCode::MAXPOOL2D: return "MAXPOOL2D";
    case OpCode::AVGPOOL2D: return "AVGPOOL2D";
    case OpCode::TRANSPOSE: return "TRANSPOSE";
    case OpCode::CONCAT: return "CONCAT";
    case OpCode::RESHAPE: return "RESHAPE";
    case OpCode::SPLIT: return "SPLIT";
    case OpCode::QUANTIZE: return "QUANTIZE";
    case OpCode::DEQUANTIZE: return "DEQUANTIZE";
    case OpCode::LAYERNORM: return "LAYERNORM";
    case OpCode::BATCHNORM: return "BATCHNORM";
    case OpCode::TENSOR_SUM: return "TENSOR_SUM";
    case OpCode::TENSOR_MEAN: return "TENSOR_MEAN";
    case OpCode::TENSOR_MAX: return "TENSOR_MAX";
    case OpCode::TENSOR_MIN: return "TENSOR_MIN";
    case OpCode::GT: return "GT";
    case OpCode::LT: return "LT";
    case OpCode::EQ: return "EQ";
    case OpCode::GE: return "GE";
    case OpCode::LE: return "LE";
    case OpCode::NE: return "NE";
    case OpCode::AND: return "AND";
    case OpCode::OR: return "OR";
    case OpCode::XOR: return "XOR";
    case OpCode::NOT: return "NOT";
    case OpCode::SYNC: return "SYNC";
    case OpCode::RETURN: return "RETURN";
    case OpCode::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

static Instruction encodeNode(const DataflowNode &n) {
  Instruction ins;
  ins.dst = reg(n.id);
  ins.level = level(n.level);
  switch (n.kind) {
    case NodeKind::Param: ins.opcode = OpCode::PARAM; break;
    case NodeKind::Const: ins.opcode = OpCode::CONST; break;
    case NodeKind::Operand: ins.opcode = OpCode::INPUT; break;
    case NodeKind::Op: ins.opcode = opcodeFor(n.op); break;
  }
  if (n.operands.size() > 0) ins.src0 = reg(n.operands[0]);
  if (n.operands.size() > 1) ins.src1 = reg(n.operands[1]);
  if (n.operands.size() > 2) ins.aux0 = reg(n.operands[2]);
  if (n.operands.size() > 3) ins.aux1 = reg(n.operands[3]);
  // IR v2 param channel: pack a 24-bit opcode-defined immediate into
  // aux0/aux1/flags (little-endian). Only valid when aux0/aux1 are not already
  // holding operand registers 3-4 — a param-bearing op with >2 operands must
  // pass its param as a tensor operand instead (PROTOCOL_SPEC 6.5).
  if (n.imm != 0) {
    if (n.imm > 0x00FFFFFFu) {
      throw std::overflow_error("IR v2 immediate exceeds 24 bits");
    }
    if (n.operands.size() > 2) {
      throw std::invalid_argument(
          "IR v2 immediate cannot coexist with >2 operand registers");
    }
    ins.aux0 = static_cast<uint8_t>(n.imm & 0xFF);
    ins.aux1 = static_cast<uint8_t>((n.imm >> 8) & 0xFF);
    ins.flags = static_cast<uint8_t>((n.imm >> 16) & 0xFF);
  }
  return ins;
}

std::vector<uint8_t> serialize(const DataflowGraph &graph) {
  validateGraph(graph);

  // Build the instruction list, inserting a SYNC barrier when the dataflow level
  // increases (so the OS executor synchronizes between parallel phases).
  std::vector<Instruction> instrs;
  int curLevel = -1;
  for (const auto &n : graph.nodes) {
    if (n.level > curLevel && curLevel >= 0) {
      Instruction sync;
      sync.opcode = OpCode::SYNC;
      sync.level = level(n.level);
      instrs.push_back(sync);
    }
    curLevel = std::max(curLevel, n.level);
    instrs.push_back(encodeNode(n));
  }
  // RETURN with the first output in dst (full output table follows the body).
  {
    Instruction ret;
    ret.opcode = OpCode::RETURN;
    if (!graph.outputs.empty()) {
      ret.dst = reg(outputValue(graph, graph.outputs[0]));
    }
    instrs.push_back(ret);
  }

  std::vector<uint8_t> out;
  out.insert(out.end(), kMagic.begin(), kMagic.end());
  put_u16(out, /*version=*/1);
  put_u16(out, /*flags=*/0);
  put_u32(out, static_cast<uint32_t>(instrs.size()));
  put_u32(out, static_cast<uint32_t>(graph.outputs.size()));
  for (const auto &ins : instrs) {
    out.push_back(static_cast<uint8_t>(ins.opcode));
    out.push_back(ins.dst);
    out.push_back(ins.src0);
    out.push_back(ins.src1);
    out.push_back(ins.aux0);
    out.push_back(ins.aux1);
    out.push_back(ins.level);
    out.push_back(ins.flags);
  }
  for (const auto &name : graph.outputs) {
    out.push_back(reg(outputValue(graph, name)));
  }
  return out;
}

std::optional<Module> deserialize(const std::vector<uint8_t> &bytes) {
  if (bytes.size() < kHeaderSize) return std::nullopt;
  if (std::memcmp(bytes.data(), kMagic.data(), 4) != 0) return std::nullopt;

  Module m;
  m.version = get_u16(&bytes[4]);
  m.flags = get_u16(&bytes[6]);
  const uint32_t instrCount = get_u32(&bytes[8]);
  const uint32_t outputCount = get_u32(&bytes[12]);

  const std::size_t instrBytes = static_cast<std::size_t>(instrCount) * kInstrSize;
  if (instrCount != 0 && instrBytes / kInstrSize != instrCount) return std::nullopt;
  if (instrBytes > bytes.size() - kHeaderSize) return std::nullopt;
  const std::size_t need = kHeaderSize + instrBytes + outputCount;
  if (need < kHeaderSize || need != bytes.size()) return std::nullopt;

  std::size_t off = kHeaderSize;
  for (uint32_t i = 0; i < instrCount; ++i, off += kInstrSize) {
    Instruction ins;
    ins.opcode = static_cast<OpCode>(bytes[off + 0]);
    ins.dst = bytes[off + 1];
    ins.src0 = bytes[off + 2];
    ins.src1 = bytes[off + 3];
    ins.aux0 = bytes[off + 4];
    ins.aux1 = bytes[off + 5];
    ins.level = bytes[off + 6];
    ins.flags = bytes[off + 7];
    m.instructions.push_back(ins);
  }
  for (uint32_t i = 0; i < outputCount; ++i) m.outputs.push_back(bytes[off + i]);
  return m;
}

}  // namespace caramel::ir
