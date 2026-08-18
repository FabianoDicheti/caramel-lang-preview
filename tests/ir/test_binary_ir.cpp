// ============================================================================
// Caramel Language - Binary IR serialization tests
// ----------------------------------------------------------------------------
// Ticket: lang_015 (IR Serialization to Binary)
// ============================================================================
#include "caramel/ir/binary_ir.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace caramel::parse;
using namespace caramel::ir;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

static DataflowGraph build(const std::string &src,
                           std::unique_ptr<caramel::ast::Program> &keep) {
  Lexer lx(src);
  Parser p(lx.tokenize());
  keep = p.parse();
  for (auto &it : keep->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      return buildDataflow(*static_cast<caramel::ast::LambdaFlow *>(it.get()));
  return {};
}

static void test_opcode_mapping() {
  CHECK(opcodeFor("matmul") == OpCode::MATMUL);
  CHECK(opcodeFor("elemwise_add") == OpCode::ADD);
  CHECK(opcodeFor("relu") == OpCode::RELU);
  CHECK(opcodeFor("not_a_real_op") == OpCode::UNKNOWN);
  CHECK(std::string(opcodeName(OpCode::MATMUL)) == "MATMUL");
  // Comparison/logic ops (x86_056): both operator-symbol and worker-mnemonic
  // spellings map to the same opcode.
  CHECK(opcodeFor(">") == OpCode::GT && opcodeFor("gt") == OpCode::GT);
  CHECK(opcodeFor("<") == OpCode::LT && opcodeFor("lt") == OpCode::LT);
  CHECK(opcodeFor("==") == OpCode::EQ && opcodeFor("eq") == OpCode::EQ);
  CHECK(opcodeFor(">=") == OpCode::GE && opcodeFor("ge") == OpCode::GE);
  CHECK(opcodeFor("<=") == OpCode::LE && opcodeFor("le") == OpCode::LE);
  CHECK(opcodeFor("!=") == OpCode::NE && opcodeFor("ne") == OpCode::NE);
  CHECK(opcodeFor("and") == OpCode::AND);
  CHECK(opcodeFor("or") == OpCode::OR);
  CHECK(opcodeFor("xor") == OpCode::XOR);
  CHECK(opcodeFor("not") == OpCode::NOT);
  CHECK(std::string(opcodeName(OpCode::GT)) == "GT");
  CHECK(std::string(opcodeName(OpCode::AND)) == "AND");
  CHECK(std::string(opcodeName(OpCode::NOT)) == "NOT");
}

static void test_serialize_roundtrip() {
  const char *src =
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=2;\n"
      "calc::lambda_flow f(a, b, c, d) {\n"
      "  @clock(1):\n"
      "    a b add r1 =\n"
      "    c d mul r2 =\n"
      "  @clock(2):\n"
      "    r1 r2 add result =\n"
      "} return result;\n";
  std::unique_ptr<caramel::ast::Program> keep;
  DataflowGraph g = build(src, keep);
  CHECK(g.nodes.size() == 7);

  std::vector<uint8_t> bytes = serialize(g);
  // header magic + 16-byte header
  CHECK(bytes.size() >= 16);
  CHECK(bytes[0] == 'C' && bytes[1] == 'R' && bytes[2] == 'M' && bytes[3] == 'L');
  // 8-byte aligned instruction region + outputs
  CHECK(((bytes.size() - 16 - 1 /*one output*/) % 8) == 0);

  auto mod = deserialize(bytes);
  CHECK(mod.has_value());
  CHECK(mod->version == 1);
  CHECK(mod->outputs.size() == 1);

  // Opcodes present: 4 PARAM, two ADD, one MUL, two SYNC (inputs->1, 1->2),
  // one RETURN.
  int params = 0, adds = 0, muls = 0, syncs = 0, rets = 0;
  for (const auto &ins : mod->instructions) {
    if (ins.opcode == OpCode::PARAM) ++params;
    if (ins.opcode == OpCode::ADD) ++adds;
    if (ins.opcode == OpCode::MUL) ++muls;
    if (ins.opcode == OpCode::SYNC) ++syncs;
    if (ins.opcode == OpCode::RETURN) ++rets;
  }
  CHECK(params == 4);
  CHECK(adds == 2);
  CHECK(muls == 1);
  CHECK(syncs == 2);   // barriers: inputs->clock1, clock1->clock2
  CHECK(rets == 1);

  // The matmul-free flow's combiner add at level 2 must carry level 2.
  bool sawLevel2 = false;
  for (const auto &ins : mod->instructions)
    if (ins.opcode == OpCode::ADD && ins.level == 2) sawLevel2 = true;
  CHECK(sawLevel2);
}

static void test_matmul_opcode_in_stream() {
  const char *src =
      "crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=1;\n"
      "calc::lambda_flow g(x, W) {\n"
      "  x W matmul out =\n"
      "} return out;\n";
  std::unique_ptr<caramel::ast::Program> keep;
  DataflowGraph g = build(src, keep);
  auto mod = deserialize(serialize(g));
  CHECK(mod.has_value());
  bool hasMatmul = false;
  for (const auto &ins : mod->instructions)
    if (ins.opcode == OpCode::MATMUL) {
      hasMatmul = true;
      CHECK(ins.src0 != ins.src1);  // two distinct operand registers (x, W)
    }
  CHECK(hasMatmul);
}

static void test_reject_garbage() {
  std::vector<uint8_t> garbage = {'X', 'Y', 'Z', 'W', 0, 0, 0, 0};
  CHECK(!deserialize(garbage).has_value());
  CHECK(!deserialize({}).has_value());
}

static void test_reject_trailing_bytes() {
  const char *src =
      "crml::quantmax=10;\ncrml::quantmin=-10;\ncrml::quantres=1;\n"
      "calc::lambda_flow g(x, W) {\n"
      "  x W matmul out =\n"
      "} return out;\n";
  std::unique_ptr<caramel::ast::Program> keep;
  std::vector<uint8_t> bytes = serialize(build(src, keep));
  bytes.push_back(0);
  CHECK(!deserialize(bytes).has_value());
}

static void test_reject_unencodable_graphs() {
  DataflowGraph tooMany;
  for (int i = 0; i < 257; ++i) {
    DataflowNode n;
    n.id = static_cast<ValueId>(tooMany.nodes.size());
    n.kind = NodeKind::Const;
    n.op = "0";
    n.level = 0;
    tooMany.nodes.push_back(n);
  }
  tooMany.outputs.push_back("last");
  tooMany.bound["last"] = 256;
  bool rejected = false;
  try {
    (void)serialize(tooMany);
  } catch (const std::overflow_error &) {
    rejected = true;
  }
  CHECK(rejected);

  DataflowGraph missingOutput;
  DataflowNode n;
  n.id = 0;
  n.kind = NodeKind::Const;
  n.op = "1";
  missingOutput.nodes.push_back(n);
  missingOutput.outputs.push_back("not_bound");
  rejected = false;
  try {
    (void)serialize(missingOutput);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);

  DataflowGraph unknownOp;
  n.kind = NodeKind::Op;
  n.op = "not_a_real_op";
  unknownOp.nodes.push_back(n);
  unknownOp.bound["out"] = 0;
  unknownOp.outputs.push_back("out");
  rejected = false;
  try {
    (void)serialize(unknownOp);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
}

// IR v2 param channel (x86_057): a node's 24-bit immediate serializes into
// aux0/aux1/flags and round-trips through deserialize + instrImm24.
static void test_imm_param_channel() {
  DataflowGraph g;
  DataflowNode a;
  a.id = 0; a.kind = NodeKind::Param; a.op = "a"; a.level = 0;
  DataflowNode op;
  op.id = 1; op.kind = NodeKind::Op; op.op = "relu"; op.operands = {0};
  op.level = 0; op.imm = 0x0A0B0Cu;  // 24-bit
  g.nodes = {a, op};
  g.bound["r"] = 1;
  g.outputs = {"r"};

  auto bytes = serialize(g);
  auto mod = deserialize(bytes);
  CHECK(mod.has_value());
  // instructions: [0]=PARAM, [1]=RELU (same level -> no SYNC), [2]=RETURN
  CHECK(mod->instructions.size() == 3);
  const auto &ri = mod->instructions[1];
  CHECK(ri.opcode == OpCode::RELU);
  CHECK(ri.aux0 == 0x0C && ri.aux1 == 0x0B && ri.flags == 0x0A);
  CHECK(instrImm24(ri) == 0x0A0B0Cu);
  // A param-free op leaves the channel zero.
  CHECK(instrImm24(mod->instructions[0]) == 0u);

  // > 24 bits is rejected.
  bool rejected = false;
  DataflowGraph tooBig = g;
  tooBig.nodes[1].imm = 0x01000000u;
  try { (void)serialize(tooBig); }
  catch (const std::overflow_error &) { rejected = true; }
  CHECK(rejected);

  // An immediate cannot coexist with a 3rd operand register.
  rejected = false;
  DataflowGraph clash;
  DataflowNode p0, p1, p2, o3;
  p0.id = 0; p0.kind = NodeKind::Param; p0.op = "x"; p0.level = 0;
  p1.id = 1; p1.kind = NodeKind::Param; p1.op = "y"; p1.level = 0;
  p2.id = 2; p2.kind = NodeKind::Param; p2.op = "z"; p2.level = 0;
  o3.id = 3; o3.kind = NodeKind::Op; o3.op = "matmul";
  o3.operands = {0, 1, 2}; o3.level = 0; o3.imm = 1;  // 3 operands + imm
  clash.nodes = {p0, p1, p2, o3};
  clash.bound["r"] = 3;
  clash.outputs = {"r"};
  try { (void)serialize(clash); }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected);
}

int main() {
  test_opcode_mapping();
  test_serialize_roundtrip();
  test_matmul_opcode_in_stream();
  test_reject_garbage();
  test_reject_trailing_bytes();
  test_reject_unencodable_graphs();
  test_imm_param_channel();
  if (g_failures == 0) {
    std::printf("OK: all binary-IR tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
