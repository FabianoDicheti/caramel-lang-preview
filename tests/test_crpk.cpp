// ============================================================================
// Caramel Language - CRPK/CRRS envelope codec tests
// ----------------------------------------------------------------------------
// Ticket: lang_040 (CRPK/CRRS Codec + Fixture Generator)
// ============================================================================
#include "caramel/proto/crpk.h"

#include "caramel/interp/interpreter.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/binary_ir.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace caramel::parse;
using namespace caramel::proto;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// The canonical fixture flow (examples/matmul_relu.crml).
static const char *kMatmulReluSrc =
    "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
    "calc::lambda_flow layer(x, w, bias) {\n"
    "    x w matmul p =\n"
    "    p bias elemwise_add s =\n"
    "    s relu result =\n"
    "} return result;\n";

static caramel::ir::DataflowGraph build(
    const std::string &src, std::unique_ptr<caramel::ast::Program> &keep) {
  Lexer lx(src);
  Parser p(lx.tokenize());
  keep = p.parse();
  for (auto &it : keep->items)
    if (it->kind == caramel::ast::NodeKind::LambdaFlow)
      return caramel::ir::buildDataflow(
          *static_cast<caramel::ast::LambdaFlow *>(it.get()));
  return {};
}

// Build the canonical CRPK request in-memory: matmul_relu IR + fixed inputs
// x=1,2,3,4  w=5,6,7,8  bias=0,0,-100,0 with quant {0, int32, -1000, 1000}.
static CrpkRequest canonicalRequest() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = build(kMatmulReluSrc, keep);
  CrpkRequest req;
  req.quant.quantres = 0;
  req.quant.dtype = kDTypeInt32;
  req.quant.quantmin = -1000;
  req.quant.quantmax = 1000;
  req.ir = caramel::ir::serialize(g);
  auto x = tensorFromValue(0, kDTypeInt32,
                           caramel::interp::Value::tensor({2, 2}, {1, 2, 3, 4}));
  auto w = tensorFromValue(1, kDTypeInt32,
                           caramel::interp::Value::tensor({2, 2}, {5, 6, 7, 8}));
  auto b = tensorFromValue(2, kDTypeInt32,
                           caramel::interp::Value::tensor({2, 2}, {0, 0, -100, 0}));
  req.inputs.push_back(*x);
  req.inputs.push_back(*w);
  req.inputs.push_back(*b);
  return req;
}

static uint32_t read_u32(const std::vector<uint8_t> &b, size_t off) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[off + i]) << (8 * i);
  return v;
}

static void test_dtype_size() {
  CHECK(dtypeSize(kDTypeInt8) == 1);
  CHECK(dtypeSize(kDTypeInt32) == 4);
  CHECK(dtypeSize(0) == 0);
  CHECK(dtypeSize(3) == 0);
}

static void test_crpk_round_trip() {
  CrpkRequest req = canonicalRequest();
  std::vector<uint8_t> bytes = serializeCrpk(req);
  auto back = parseCrpk(bytes);
  CHECK(back.has_value());
  CHECK(*back == req);
  // The embedded IR is byte-identical and still deserializes as a module.
  CHECK(back->ir == req.ir);
  CHECK(caramel::ir::deserialize(back->ir).has_value());
}

static void test_crrs_round_trip() {
  CrrsResponse resp;
  resp.status = 0;
  resp.outputs.push_back(*tensorFromValue(
      0, kDTypeInt32, caramel::interp::Value::tensor({2, 2}, {19, 22, 0, 50})));
  std::vector<uint8_t> bytes = buildCrrs(resp);
  auto back = parseCrrs(bytes);
  CHECK(back.has_value());
  CHECK(*back == resp);
}

// Golden byte layout of the canonical CRPK at fixed offsets (PROTOCOL 6.5).
static void test_crpk_golden_offsets() {
  CrpkRequest req = canonicalRequest();
  std::vector<uint8_t> b = serializeCrpk(req);

  CHECK(b.size() == 24 + req.ir.size() + 3 * (4 + 8 + 16));
  CHECK(b[0] == 'C' && b[1] == 'R' && b[2] == 'P' && b[3] == 'K');
  CHECK(b[4] == 1 && b[5] == 0);                    // version u16 = 1
  CHECK(b[6] == 0 && b[7] == 0);                    // flags u16 = 0
  CHECK(read_u32(b, 8) == req.ir.size());           // ir_size
  CHECK(read_u32(b, 12) == 3);                      // tensor_count
  CHECK(b[16] == 0);                                // quantres
  CHECK(b[17] == 2);                                // dtype int32
  CHECK(b[18] == 0 && b[19] == 0);                  // reserved
  CHECK(b[20] == 0x18 && b[21] == 0xFC);            // quantmin -1000
  CHECK(b[22] == 0xE8 && b[23] == 0x03);            // quantmax 1000
  // Embedded IR module starts with its own "CRML" magic.
  CHECK(b[24] == 'C' && b[25] == 'R' && b[26] == 'M' && b[27] == 'L');

  // First tensor entry (x) right after the IR blob.
  const size_t e = 24 + req.ir.size();
  CHECK(b[e + 0] == 0);                             // param_index
  CHECK(b[e + 1] == 2);                             // dtype int32
  CHECK(b[e + 2] == 2);                             // rank
  CHECK(b[e + 3] == 0);                             // reserved
  CHECK(read_u32(b, e + 4) == 2 && read_u32(b, e + 8) == 2);  // dims 2x2
  CHECK(read_u32(b, e + 12) == 1 && read_u32(b, e + 16) == 2 &&
        read_u32(b, e + 20) == 3 && read_u32(b, e + 24) == 4);  // 1,2,3,4 LE

  // Third tensor entry (bias) carries -100 as a little-endian int32.
  const size_t e3 = e + 2 * 28 + 12;                // skip x, w, bias header+dims
  CHECK(read_u32(b, e3 + 8) == 0xFFFFFF9Cu);        // bias[2] == -100
}

// The golden CRRS equals the local interpreter's result for the same inputs.
static void test_crrs_matches_interpreter() {
  std::unique_ptr<caramel::ast::Program> keep;
  auto g = build(kMatmulReluSrc, keep);
  caramel::interp::Interpreter vm;
  vm.add_evaluator(caramel::interp::matrixOpEvaluator());
  vm.set_input("x", caramel::interp::Value::tensor({2, 2}, {1, 2, 3, 4}));
  vm.set_input("w", caramel::interp::Value::tensor({2, 2}, {5, 6, 7, 8}));
  vm.set_input("bias", caramel::interp::Value::tensor({2, 2}, {0, 0, -100, 0}));
  auto r = vm.run(g);
  CHECK(r.ok());
  const auto &out = r.outputs["result"];
  CHECK((out.dims == std::vector<int64_t>{2, 2}));
  CHECK((out.data == std::vector<int64_t>{19, 22, 0, 50}));

  CrrsResponse resp;
  resp.outputs.push_back(*tensorFromValue(0, kDTypeInt32, out));
  auto back = parseCrrs(buildCrrs(resp));
  CHECK(back.has_value());
  CHECK(back->status == 0);
  CHECK(back->outputs.size() == 1);
  auto v = valueFromTensor(back->outputs[0]);
  CHECK(v.has_value());
  CHECK((v->dims == std::vector<int64_t>{2, 2}));
  CHECK((v->data == std::vector<int64_t>{19, 22, 0, 50}));
}

static void test_value_adapters() {
  // Scalars (rank 0) and rank > 4 are not representable on the wire.
  CHECK(!tensorFromValue(0, kDTypeInt32, caramel::interp::Value::scalar(7)).has_value());
  CHECK(!tensorFromValue(0, kDTypeInt32,
                         caramel::interp::Value::tensor({1, 1, 1, 1, 1}, {9}))
             .has_value());
  // int8 range checking.
  CHECK(tensorFromValue(0, kDTypeInt8,
                        caramel::interp::Value::tensor({2}, {-128, 127}))
            .has_value());
  CHECK(!tensorFromValue(0, kDTypeInt8,
                         caramel::interp::Value::tensor({2}, {-129, 0}))
             .has_value());
  CHECK(!tensorFromValue(0, kDTypeInt32,
                         caramel::interp::Value::tensor({1}, {int64_t{1} << 40}))
             .has_value());
  // int8 round trip sign-extends.
  auto t = tensorFromValue(3, kDTypeInt8,
                           caramel::interp::Value::tensor({2, 2}, {-1, -128, 0, 127}));
  CHECK(t.has_value());
  auto v = valueFromTensor(*t);
  CHECK(v.has_value());
  CHECK((v->data == std::vector<int64_t>{-1, -128, 0, 127}));
}

static void test_crpk_rejects_malformed() {
  const std::vector<uint8_t> good = serializeCrpk(canonicalRequest());
  CHECK(parseCrpk(good).has_value());

  // Truncation at every byte boundary (covers header / quant / IR / every
  // tensor-entry section edge) must fail, never crash.
  for (size_t n = 0; n < good.size(); ++n) {
    std::vector<uint8_t> cut(good.begin(), good.begin() + n);
    CHECK(!parseCrpk(cut).has_value());
  }
  // Trailing garbage.
  {
    std::vector<uint8_t> b = good;
    b.push_back(0);
    CHECK(!parseCrpk(b).has_value());
  }
  // Bad magic (the bad_magic.crpk mutation).
  {
    std::vector<uint8_t> b = good;
    b[0] = 'X';
    CHECK(!parseCrpk(b).has_value());
  }
  // Unsupported version / nonzero reserved flags.
  {
    std::vector<uint8_t> b = good;
    b[4] = 2;
    CHECK(!parseCrpk(b).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    b[6] = 1;
    CHECK(!parseCrpk(b).has_value());
  }
  // Unknown quant dtype.
  {
    std::vector<uint8_t> b = good;
    b[17] = 9;
    CHECK(!parseCrpk(b).has_value());
  }
  // ir_size larger than the whole buffer.
  {
    std::vector<uint8_t> b = good;
    b[8] = 0xFF; b[9] = 0xFF; b[10] = 0xFF; b[11] = 0xFF;
    CHECK(!parseCrpk(b).has_value());
  }
  const size_t irSize = read_u32(good, 8);
  const size_t entry = 24 + irSize;
  // Tensor rank 0 and 5 (rank must be 1..4).
  {
    std::vector<uint8_t> b = good;
    b[entry + 2] = 0;
    CHECK(!parseCrpk(b).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    b[entry + 2] = 5;
    CHECK(!parseCrpk(b).has_value());
  }
  // Nonzero reserved byte in a tensor entry.
  {
    std::vector<uint8_t> b = good;
    b[entry + 3] = 1;
    CHECK(!parseCrpk(b).has_value());
  }
  // Unknown tensor dtype.
  {
    std::vector<uint8_t> b = good;
    b[entry + 1] = 7;
    CHECK(!parseCrpk(b).has_value());
  }
  // Zero dim.
  {
    std::vector<uint8_t> b = good;
    b[entry + 4] = 0; b[entry + 5] = 0; b[entry + 6] = 0; b[entry + 7] = 0;
    CHECK(!parseCrpk(b).has_value());
  }
  // Dims overflow: 0xFFFFFFFF x 0xFFFFFFFF elements x 4 bytes must not wrap
  // into an accepted small size.
  {
    std::vector<uint8_t> b = good;
    for (int i = 0; i < 8; ++i) b[entry + 4 + i] = 0xFF;
    CHECK(!parseCrpk(b).has_value());
  }
  // The bad_shape mutation: x declared 2x3 while the data stays 2x2.
  {
    std::vector<uint8_t> b = good;
    b[entry + 8] = 3;
    CHECK(!parseCrpk(b).has_value());
  }
  // Non-dense param_index.
  {
    std::vector<uint8_t> b = good;
    b[entry + 0] = 1;
    CHECK(!parseCrpk(b).has_value());
  }
  // tensor_count larger than the entries present.
  {
    std::vector<uint8_t> b = good;
    b[12] = 4;
    CHECK(!parseCrpk(b).has_value());
  }
}

static void test_crrs_rejects_malformed() {
  CrrsResponse resp;
  resp.status = 0;
  resp.outputs.push_back(*tensorFromValue(
      0, kDTypeInt32, caramel::interp::Value::tensor({2, 2}, {19, 22, 0, 50})));
  const std::vector<uint8_t> good = buildCrrs(resp);
  CHECK(parseCrrs(good).has_value());

  CHECK(!parseCrrs({}).has_value());
  for (size_t n = 0; n < good.size(); ++n) {
    std::vector<uint8_t> cut(good.begin(), good.begin() + n);
    CHECK(!parseCrrs(cut).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    b[0] = 'X';
    CHECK(!parseCrrs(b).has_value());
  }
  {  // CRPK magic is not CRRS magic
    std::vector<uint8_t> b = good;
    b[2] = 'P'; b[3] = 'K';
    CHECK(!parseCrrs(b).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    b[4] = 9;  // version
    CHECK(!parseCrrs(b).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    b[12 + 2] = 5;  // rank 5
    CHECK(!parseCrrs(b).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    b[12 + 2] = 0;  // rank 0
    CHECK(!parseCrrs(b).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    for (int i = 0; i < 8; ++i) b[12 + 4 + i] = 0xFF;  // huge dims
    CHECK(!parseCrrs(b).has_value());
  }
  {
    std::vector<uint8_t> b = good;
    b.push_back(0);  // trailing byte
    CHECK(!parseCrrs(b).has_value());
  }
  // A nonzero status with no tensors is a valid error response.
  {
    CrrsResponse err;
    err.status = 3;
    auto back = parseCrrs(buildCrrs(err));
    CHECK(back.has_value());
    CHECK(back->status == 3);
    CHECK(back->outputs.empty());
  }
}

static void test_serialize_rejects_unencodable() {
  // Rank 0 entry.
  CrpkRequest req = canonicalRequest();
  req.inputs[0].dims.clear();
  bool rejected = false;
  try {
    (void)serializeCrpk(req);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);

  // Data size not matching dims.
  req = canonicalRequest();
  req.inputs[1].data.pop_back();
  rejected = false;
  try {
    (void)serializeCrpk(req);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);

  // Non-dense param_index ordering.
  req = canonicalRequest();
  req.inputs[2].param_index = 7;
  rejected = false;
  try {
    (void)serializeCrpk(req);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);

  // buildCrrs applies the same entry validation.
  CrrsResponse resp;
  resp.outputs.push_back(TensorEntry{});  // rank 0, no data
  rejected = false;
  try {
    (void)buildCrrs(resp);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
}

// Full pipeline twice -> identical bytes (fixture determinism).
static void test_determinism() {
  std::vector<uint8_t> a = serializeCrpk(canonicalRequest());
  std::vector<uint8_t> b = serializeCrpk(canonicalRequest());
  CHECK(a == b);

  CrrsResponse resp;
  resp.outputs.push_back(*tensorFromValue(
      0, kDTypeInt32, caramel::interp::Value::tensor({2, 2}, {19, 22, 0, 50})));
  CHECK(buildCrrs(resp) == buildCrrs(resp));
}

int main() {
  test_dtype_size();
  test_crpk_round_trip();
  test_crrs_round_trip();
  test_crpk_golden_offsets();
  test_crrs_matches_interpreter();
  test_value_adapters();
  test_crpk_rejects_malformed();
  test_crrs_rejects_malformed();
  test_serialize_rejects_unencodable();
  test_determinism();
  if (g_failures == 0) {
    std::printf("OK: all CRPK/CRRS codec tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
