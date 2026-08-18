// ============================================================================
// Caramel Language - crpk-gen: golden CRPK/CRRS fixture generator
// ----------------------------------------------------------------------------
// Ticket: lang_040 (CRPK/CRRS Codec + Fixture Generator)
// Compiles a .crml flow through the existing front-end (lexer -> parser ->
// dataflow -> binary IR), packages fixed inputs into a CRPK request envelope,
// runs the local CPU interpreter to compute the expected outputs, and writes
// the conformance fixtures defined in priority/fixtures_spec.md:
//   <stem>.crpk         good request (header + quant + IR + input tensors)
//   <stem>.crrs         expected response (status 0 + output tensors)
//   <stem>.json         human-readable description (not parsed by tests)
//   bad_magic.crpk      good file with byte 0 corrupted ('X')
//   bad_truncated.crpk  good file with the last 8 bytes dropped (mid-tensor)
//   bad_shape.crpk      first input's dims patched 2x2 -> 2x3 (dims only)
//   unsupported_op.crpk first MATMUL opcode in the IR overwritten with CONV2D
//
// Output is byte-deterministic: no timestamps, no randomness; the bad_* files
// are derived from the good byte buffer by scripted mutation.
//
// Usage:
//   crpk-gen <file.crml> --out DIR [--flow NAME] [--in name=VALUE]...
//            [--mutate bad_magic|bad_truncated|bad_shape|unsupported_op]
// Inputs default to the canonical matmul_relu case when no --in is given:
//   x=2x2:1,2,3,4  w=2x2:5,6,7,8  bias=2x2:0,0,-100,0
// With --mutate only that one derived fixture is written; by default one run
// generates all 7 files.
// ============================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "caramel/interp/interpreter.h"
#include "caramel/interp/lambda.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/binary_ir.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"
#include "caramel/proto/crpk.h"
#include "caramel/types/type.h"

using namespace caramel;

namespace {

std::vector<std::string> split(const std::string &s, char d) {
  std::vector<std::string> out;
  std::string cur;
  std::stringstream ss(s);
  while (std::getline(ss, cur, d)) out.push_back(cur);
  return out;
}

bool parse_i64(const std::string &s, int64_t &out) {
  if (s.empty()) return false;
  size_t pos = 0;
  try {
    out = std::stoll(s, &pos);
  } catch (const std::exception &) {
    return false;
  }
  return pos == s.size();
}

// Parse "name=2x2:1,2,3,4" (same tensor syntax as caramel-run --in).
bool parse_input(const std::string &arg, std::string &name, interp::Value &value) {
  auto eq = arg.find('=');
  if (eq == std::string::npos) return false;
  name = arg.substr(0, eq);
  if (name.empty()) return false;
  std::string rhs = arg.substr(eq + 1);
  auto colon = rhs.find(':');
  if (colon == std::string::npos) return false;  // fixtures are tensors only
  std::vector<int64_t> dims;
  for (auto &d : split(rhs.substr(0, colon), 'x')) {
    int64_t dim = 0;
    if (!parse_i64(d, dim) || dim <= 0) return false;
    dims.push_back(dim);
  }
  std::vector<int64_t> data;
  for (auto &e : split(rhs.substr(colon + 1), ',')) {
    int64_t elem = 0;
    if (!parse_i64(e, elem)) return false;
    data.push_back(elem);
  }
  int64_t expected = 1;
  for (int64_t d : dims) expected *= d;
  if (dims.empty() || data.size() != static_cast<size_t>(expected)) return false;
  value = interp::Value::tensor(std::move(dims), std::move(data));
  return true;
}

bool write_file(const std::string &path, const std::vector<uint8_t> &bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f.write(reinterpret_cast<const char *>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(f);
}

bool write_text(const std::string &path, const std::string &text) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f << text;
  return static_cast<bool>(f);
}

// --- Scripted mutations over the good CRPK byte buffer ---------------------
// Byte 0 corrupted -> workers must fail BAD_ENVELOPE.
std::vector<uint8_t> mutate_bad_magic(std::vector<uint8_t> b) {
  b[0] = 'X';
  return b;
}

// Cut mid-tensor-data (the last entry's data is 16 bytes) -> BAD_ENVELOPE.
std::vector<uint8_t> mutate_bad_truncated(std::vector<uint8_t> b) {
  if (b.size() < 8) return b;
  b.resize(b.size() - 8);
  return b;
}

// First input declares one extra column (2x2 -> 2x3); dims field only, data
// and IR untouched -> SHAPE_MISMATCH.
bool mutate_bad_shape(std::vector<uint8_t> &b) {
  if (b.size() < 24) return false;
  uint32_t irSize = 0;
  for (int i = 0; i < 4; ++i) irSize |= static_cast<uint32_t>(b[8 + i]) << (8 * i);
  const size_t entry = 24 + static_cast<size_t>(irSize);
  if (entry + 12 > b.size()) return false;
  if (b[entry + 2] < 2) return false;  // need rank >= 2 to patch dims[1]
  const size_t dim1 = entry + 4 + 4;   // entry header + dims[0]
  b[dim1 + 0] = 3;                     // dims[1] = 3 (u32 little-endian)
  b[dim1 + 1] = 0;
  b[dim1 + 2] = 0;
  b[dim1 + 3] = 0;
  return true;
}

// Overwrite the first MATMUL opcode inside the embedded IR with CONV2D (0x30);
// envelope and IR header stay valid -> UNSUPPORTED_OP on M1 workers.
bool mutate_unsupported_op(std::vector<uint8_t> &b) {
  if (b.size() < 24) return false;
  uint32_t irSize = 0;
  for (int i = 0; i < 4; ++i) irSize |= static_cast<uint32_t>(b[8 + i]) << (8 * i);
  const size_t irStart = 24;
  if (irSize < 16 || irStart + irSize > b.size()) return false;
  uint32_t instrCount = 0;
  for (int i = 0; i < 4; ++i)
    instrCount |= static_cast<uint32_t>(b[irStart + 8 + i]) << (8 * i);
  for (uint32_t i = 0; i < instrCount; ++i) {
    const size_t off = irStart + 16 + static_cast<size_t>(i) * 8;
    if (off + 8 > irStart + irSize) return false;
    if (b[off] == static_cast<uint8_t>(ir::OpCode::MATMUL)) {
      // Overwrite with a reserved opcode the worker does not implement, to
      // exercise capability gating. QUANTIZE (0x50) stays reserved; CONV2D
      // (0x30) is now a supported op (x86_059) and no longer unsupported.
      b[off] = static_cast<uint8_t>(ir::OpCode::QUANTIZE);
      return true;
    }
  }
  return false;  // no MATMUL to overwrite
}

std::string json_tensor(const std::string &name, const proto::TensorEntry &entry,
                        const interp::Value &v) {
  std::ostringstream os;
  os << "{\"name\": \"" << name << "\", \"param_index\": "
     << static_cast<int>(entry.param_index) << ", \"dtype\": \""
     << (entry.dtype == proto::kDTypeInt8 ? "int8" : "int32") << "\", \"dims\": [";
  for (size_t i = 0; i < entry.dims.size(); ++i) {
    if (i) os << ", ";
    os << entry.dims[i];
  }
  os << "], \"values\": [";
  for (size_t i = 0; i < v.data.size(); ++i) {
    if (i) os << ", ";
    os << v.data[i];
  }
  os << "]}";
  return os.str();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: crpk-gen <file.crml> --out DIR [--flow NAME] "
                 "[--in name=VALUE]... [--mutate NAME]\n";
    return 2;
  }
  std::string path = argv[1];
  std::string out_dir;
  std::string flow_name;
  std::string mutate;
  std::map<std::string, interp::Value> given_inputs;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) { out_dir = argv[++i]; }
    else if (a == "--flow" && i + 1 < argc) { flow_name = argv[++i]; }
    else if (a == "--mutate" && i + 1 < argc) { mutate = argv[++i]; }
    else if (a == "--in" && i + 1 < argc) {
      std::string n; interp::Value v;
      if (!parse_input(argv[++i], n, v)) { std::cerr << "bad --in: " << argv[i] << "\n"; return 2; }
      given_inputs[n] = std::move(v);
    } else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
  }
  if (out_dir.empty()) { std::cerr << "error: --out DIR is required\n"; return 2; }
  if (!mutate.empty() && mutate != "bad_magic" && mutate != "bad_truncated" &&
      mutate != "bad_shape" && mutate != "unsupported_op") {
    std::cerr << "error: unknown --mutate " << mutate << "\n";
    return 2;
  }

  // Canonical fixture inputs (fixtures_spec.md) when none are given.
  if (given_inputs.empty()) {
    given_inputs["x"] = interp::Value::tensor({2, 2}, {1, 2, 3, 4});
    given_inputs["w"] = interp::Value::tensor({2, 2}, {5, 6, 7, 8});
    given_inputs["bias"] = interp::Value::tensor({2, 2}, {0, 0, -100, 0});
  }

  // Fixture stem from the source file name (matmul_relu.crml -> matmul_relu).
  std::string stem = path;
  if (auto slash = stem.find_last_of('/'); slash != std::string::npos)
    stem = stem.substr(slash + 1);
  if (auto dot = stem.find_last_of('.'); dot != std::string::npos)
    stem = stem.substr(0, dot);

  std::ifstream f(path);
  if (!f) { std::cerr << "error: cannot open " << path << "\n"; return 1; }
  std::stringstream buf; buf << f.rdbuf();

  parse::Lexer lexer(buf.str());
  parse::Parser parser(lexer.tokenize());
  auto program = parser.parse();
  if (!parser.ok()) {
    for (const auto &e : parser.errors())
      std::cerr << "parse error [" << e.loc.line << ":" << e.loc.column << "] "
                << e.message << "\n";
    return 1;
  }

  // Quant block from the crml::quant* directives.
  int64_t quantmax = 0, quantmin = 0, quantres = 0;
  bool have_max = false, have_min = false, have_res = false;
  for (auto &it : program->items) {
    if (it->kind != ast::NodeKind::Directive) continue;
    auto *d = static_cast<ast::Directive *>(it.get());
    if (d->dkind != ast::DirectiveKind::Quant) continue;
    int64_t v = 0;
    if (!parse_i64(d->value, v)) continue;
    if (d->key == "quantmax") { quantmax = v; have_max = true; }
    if (d->key == "quantmin") { quantmin = v; have_min = true; }
    if (d->key == "quantres" || d->key == "decimal_places") { quantres = v; have_res = true; }
  }
  if (!have_max || !have_min || !have_res) {
    std::cerr << "error: program must define integer quantmax/quantmin/quantres\n";
    return 1;
  }
  if (quantres < 0 || quantres > 255 ||
      quantmin < std::numeric_limits<int16_t>::min() ||
      quantmin > std::numeric_limits<int16_t>::max() ||
      quantmax < std::numeric_limits<int16_t>::min() ||
      quantmax > std::numeric_limits<int16_t>::max()) {
    std::cerr << "error: quant directives do not fit the CRPK quant block\n";
    return 1;
  }
  proto::QuantBlock quant;
  quant.quantres = static_cast<uint8_t>(quantres);
  quant.quantmin = static_cast<int16_t>(quantmin);
  quant.quantmax = static_cast<int16_t>(quantmax);
  const auto qp = types::QuantParams::from_range(
      static_cast<double>(quantmax), static_cast<double>(quantmin),
      static_cast<int>(quantres));
  quant.dtype = qp.bit_width <= 8 ? proto::kDTypeInt8 : proto::kDTypeInt32;

  const ast::LambdaFlow *flow = nullptr;
  for (auto &it : program->items)
    if (it->kind == ast::NodeKind::LambdaFlow) {
      auto *lf = static_cast<ast::LambdaFlow *>(it.get());
      if (flow_name.empty() || lf->name == flow_name) { flow = lf; break; }
    }
  if (!flow) { std::cerr << "error: no lambda_flow"
                         << (flow_name.empty() ? "" : " named '" + flow_name + "'")
                         << " found\n"; return 1; }

  // Compile to binary IR and package the inputs (declaration order = slot).
  auto graph = ir::buildDataflow(*flow);
  proto::CrpkRequest req;
  req.quant = quant;
  try {
    req.ir = ir::serialize(graph);
  } catch (const std::exception &e) {
    std::cerr << "error: IR serialization failed: " << e.what() << "\n";
    return 1;
  }

  std::vector<std::pair<std::string, interp::Value>> ordered_inputs;
  for (size_t i = 0; i < flow->params.size(); ++i) {
    const std::string &pname = flow->params[i];
    auto it = given_inputs.find(pname);
    if (it == given_inputs.end()) {
      std::cerr << "error: no input for flow parameter '" << pname << "'\n";
      return 1;
    }
    auto entry = proto::tensorFromValue(static_cast<uint8_t>(i), quant.dtype,
                                        it->second);
    if (!entry) {
      std::cerr << "error: input '" << pname << "' cannot be encoded\n";
      return 1;
    }
    req.inputs.push_back(std::move(*entry));
    ordered_inputs.emplace_back(pname, it->second);
  }

  // Local interpreter run -> the expected CRRS outputs.
  interp::Interpreter vm;
  vm.add_evaluator(interp::matrixOpEvaluator());
  vm.add_evaluator(interp::lambdaOpEvaluator());
  for (auto &kv : ordered_inputs) vm.set_input(kv.first, kv.second);
  auto result = vm.run(graph);
  if (!result.ok()) {
    std::cerr << "runtime error: " << result.error->message << "\n";
    return 1;
  }

  proto::CrrsResponse resp;
  resp.status = 0;
  std::vector<std::pair<std::string, interp::Value>> ordered_outputs;
  for (size_t i = 0; i < flow->returns.size(); ++i) {
    const std::string &oname = flow->returns[i];
    auto it = result.outputs.find(oname);
    if (it == result.outputs.end()) {
      std::cerr << "error: interpreter produced no output '" << oname << "'\n";
      return 1;
    }
    auto entry = proto::tensorFromValue(static_cast<uint8_t>(i), quant.dtype,
                                        it->second);
    if (!entry) {
      std::cerr << "error: output '" << oname << "' cannot be encoded\n";
      return 1;
    }
    resp.outputs.push_back(std::move(*entry));
    ordered_outputs.emplace_back(oname, it->second);
  }

  std::vector<uint8_t> crpk;
  std::vector<uint8_t> crrs;
  try {
    crpk = proto::serializeCrpk(req);
    crrs = proto::buildCrrs(resp);
  } catch (const std::exception &e) {
    std::cerr << "error: envelope serialization failed: " << e.what() << "\n";
    return 1;
  }

  // Sanity: the codec must round-trip its own bytes.
  if (!proto::parseCrpk(crpk) || !proto::parseCrrs(crrs)) {
    std::cerr << "error: generated envelope does not parse back\n";
    return 1;
  }

  auto out_path = [&](const std::string &name) { return out_dir + "/" + name; };
  auto emit = [&](const std::string &name, const std::vector<uint8_t> &bytes) {
    if (!write_file(out_path(name), bytes)) {
      std::cerr << "error: cannot write " << out_path(name) << "\n";
      return false;
    }
    std::cout << name << "  " << bytes.size() << " bytes\n";
    return true;
  };

  // Derived (mutated) fixtures from the good byte buffer.
  std::vector<uint8_t> bad_magic = mutate_bad_magic(crpk);
  std::vector<uint8_t> bad_truncated = mutate_bad_truncated(crpk);
  std::vector<uint8_t> bad_shape = crpk;
  if (!mutate_bad_shape(bad_shape)) {
    std::cerr << "error: cannot derive bad_shape (first input rank < 2?)\n";
    return 1;
  }
  std::vector<uint8_t> unsupported = crpk;
  if (!mutate_unsupported_op(unsupported)) {
    std::cerr << "error: cannot derive unsupported_op (no MATMUL in IR?)\n";
    return 1;
  }

  if (!mutate.empty()) {
    if (mutate == "bad_magic") return emit("bad_magic.crpk", bad_magic) ? 0 : 1;
    if (mutate == "bad_truncated") return emit("bad_truncated.crpk", bad_truncated) ? 0 : 1;
    if (mutate == "bad_shape") return emit("bad_shape.crpk", bad_shape) ? 0 : 1;
    return emit("unsupported_op.crpk", unsupported) ? 0 : 1;
  }

  // Human-readable description (for debugging; not parsed by tests).
  auto mod = ir::deserialize(req.ir);
  std::ostringstream js;
  js << "{\n";
  js << "  \"flow\": \"" << flow->name << "\",\n";
  js << "  \"source\": \"" << path << "\",\n";
  js << "  \"quant\": {\"quantres\": " << static_cast<int>(quant.quantres)
     << ", \"dtype\": \"" << (quant.dtype == proto::kDTypeInt8 ? "int8" : "int32")
     << "\", \"quantmin\": " << quant.quantmin
     << ", \"quantmax\": " << quant.quantmax << "},\n";
  js << "  \"ir\": {\"size_bytes\": " << req.ir.size()
     << ", \"instruction_count\": " << (mod ? mod->instructions.size() : 0)
     << ", \"disassembly\": [";
  if (mod) {
    for (size_t i = 0; i < mod->instructions.size(); ++i) {
      const auto &ins = mod->instructions[i];
      if (i) js << ", ";
      js << "\"" << ir::opcodeName(ins.opcode)
         << " dst=" << static_cast<int>(ins.dst)
         << " src0=" << static_cast<int>(ins.src0)
         << " src1=" << static_cast<int>(ins.src1)
         << " level=" << static_cast<int>(ins.level) << "\"";
    }
  }
  js << "]},\n";
  js << "  \"inputs\": [\n";
  for (size_t i = 0; i < req.inputs.size(); ++i) {
    js << "    " << json_tensor(ordered_inputs[i].first, req.inputs[i],
                                ordered_inputs[i].second)
       << (i + 1 < req.inputs.size() ? ",\n" : "\n");
  }
  js << "  ],\n";
  js << "  \"expected_outputs\": [\n";
  for (size_t i = 0; i < resp.outputs.size(); ++i) {
    js << "    " << json_tensor(ordered_outputs[i].first, resp.outputs[i],
                                ordered_outputs[i].second)
       << (i + 1 < resp.outputs.size() ? ",\n" : "\n");
  }
  js << "  ]\n";
  js << "}\n";

  if (!emit(stem + ".crpk", crpk)) return 1;
  if (!emit(stem + ".crrs", crrs)) return 1;
  if (!write_text(out_path(stem + ".json"), js.str())) {
    std::cerr << "error: cannot write " << out_path(stem + ".json") << "\n";
    return 1;
  }
  std::cout << stem << ".json  " << js.str().size() << " bytes\n";
  if (!emit("bad_magic.crpk", bad_magic)) return 1;
  if (!emit("bad_truncated.crpk", bad_truncated)) return 1;
  if (!emit("bad_shape.crpk", bad_shape)) return 1;
  if (!emit("unsupported_op.crpk", unsupported)) return 1;
  return 0;
}
