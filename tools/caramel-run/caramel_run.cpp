// ============================================================================
// Caramel Language - caramel-run: the interpreter CLI
// ----------------------------------------------------------------------------
// Ticket: lang_020 (Interpreter CLI), lang_043 (Remote Execution, --device)
// Loads a .crml program, parses it, builds the SSA dataflow graph for a flow, and
// executes it on the CPU interpreter (simulation mode), printing the results.
// With --device the compiled flow is instead shipped to a CDP worker
// (PROTOCOL_SPEC.md) via POST /api/execute?mode=sync and the decoded CRRS
// outputs are printed in the exact same format as a local run.
//
// Usage:
//   caramel-run <file.crml> [--flow NAME] [--in name=VALUE] ...
//               [--device http://HOST:PORT] [--device-user USER]
//               [--device-pass PASS|env:NAME] [--device-token TOK|env:NAME]
//               [--verify]
// Inputs:
//   --in a=5             scalar 5
//   --in m=2x2:1,2,3,4   tensor of shape 2x2 with row-major data
// Remote mode (lang_043):
//   --device URL         execute on a CDP worker instead of locally; the
//                        default (no --device) local behavior is unchanged
//   --device-user/-pass  login credentials (pass falls back to the
//                        CARAMEL_DEVICE_PASS environment variable)
//   --device-token       pre-issued session token (mutually exclusive with
//                        user/pass); secrets accept the "env:NAME" spelling
//   --verify             run BOTH locally and remotely, compare the decoded
//                        outputs exactly, print PASS (exit 0) or MISMATCH
//                        (exit 1) - the M1 acceptance mechanism
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "caramel/analysis/matrix_profile.h"
#include "caramel/interp/generators.h"
#include "caramel/interp/interpreter.h"
#include "caramel/interp/lambda.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/interp/activation.h"
#include "caramel/interp/norm.h"
#include "caramel/interp/shape.h"
#include "caramel/interp/spatial.h"
#include "caramel/ir/binary_ir.h"
#include "caramel/ir/dataflow.h"
#include "caramel/net/device.h"
#include "caramel/net/dispatch.h"
#include "caramel/net/remote_run.h"
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

bool shape_numel(const std::vector<int64_t> &dims, size_t &numel) {
  if (dims.empty()) return false;
  numel = 1;
  for (int64_t dim : dims) {
    if (dim <= 0) return false;
    const auto d = static_cast<size_t>(dim);
    if (numel > std::numeric_limits<size_t>::max() / d) return false;
    numel *= d;
  }
  return true;
}

// Parse "name=VALUE" where VALUE is "5" (scalar) or "2x2:1,2,3,4" (tensor).
bool parse_input(const std::string &arg, std::string &name, interp::Value &value) {
  auto eq = arg.find('=');
  if (eq == std::string::npos) return false;
  name = arg.substr(0, eq);
  if (name.empty()) return false;
  std::string rhs = arg.substr(eq + 1);
  auto colon = rhs.find(':');
  if (colon == std::string::npos) {  // scalar
    int64_t scalar = 0;
    if (!parse_i64(rhs, scalar)) return false;
    value = interp::Value::scalar(scalar);
    return true;
  }
  std::vector<int64_t> dims;
  for (auto &d : split(rhs.substr(0, colon), 'x')) {
    int64_t dim = 0;
    if (!parse_i64(d, dim)) return false;
    dims.push_back(dim);
  }
  std::vector<int64_t> data;
  for (auto &e : split(rhs.substr(colon + 1), ',')) {
    int64_t elem = 0;
    if (!parse_i64(e, elem)) return false;
    data.push_back(elem);
  }
  size_t expected = 0;
  if (!shape_numel(dims, expected) || data.size() != expected) return false;
  value = interp::Value::tensor(std::move(dims), std::move(data));
  return true;
}

// Convert a parsed `in::name = <literal>;` value (a scalar NumberLiteral or a
// nested TensorLiteral) into an interp::Value, mirroring --in. Row-major;
// integer elements (the wire format is integer-quantized).
bool literal_to_value(const ast::Expr *e, interp::Value &out) {
  if (!e) return false;
  if (e->kind == ast::NodeKind::NumberLiteral) {
    const auto *n = static_cast<const ast::NumberLiteral *>(e);
    out = interp::Value::scalar(std::strtoll(n->lexeme.c_str(), nullptr, 10));
    return true;
  }
  if (e->kind != ast::NodeKind::TensorLiteral) return false;
  std::vector<int64_t> dims, data;
  bool ok = true;
  std::function<void(const ast::Expr *, int)> walk = [&](const ast::Expr *node, int depth) {
    if (node->kind == ast::NodeKind::TensorLiteral) {
      const auto *tl = static_cast<const ast::TensorLiteral *>(node);
      if (static_cast<int>(dims.size()) <= depth) dims.push_back(static_cast<int64_t>(tl->elements.size()));
      for (const auto &el : tl->elements) walk(el.get(), depth + 1);
    } else if (node->kind == ast::NodeKind::NumberLiteral) {
      data.push_back(std::strtoll(static_cast<const ast::NumberLiteral *>(node)->lexeme.c_str(), nullptr, 10));
    } else {
      ok = false;
    }
  };
  walk(e, 0);
  if (!ok) return false;
  out = interp::Value::tensor(std::move(dims), std::move(data));
  return true;
}

// --- Tensor constructors (host-side initializers) --------------------------
// RPN postfix generators usable in `in::name = <gen>;` position, evaluated
// here at load time (they never enter the dataflow graph). See op_registry.cpp
// for the registered arities.

bool init_scalar(const ast::Expr *e, int64_t &out) {
  if (!e || e->kind != ast::NodeKind::NumberLiteral) return false;
  out = std::strtoll(static_cast<const ast::NumberLiteral *>(e)->lexeme.c_str(),
                     nullptr, 10);
  return true;
}

bool init_vector(const ast::Expr *e, std::vector<int64_t> &out) {
  if (!e || e->kind != ast::NodeKind::TensorLiteral) return false;
  const auto *tl = static_cast<const ast::TensorLiteral *>(e);
  for (const auto &el : tl->elements) {
    if (el->kind != ast::NodeKind::NumberLiteral) return false;
    out.push_back(std::strtoll(
        static_cast<const ast::NumberLiteral *>(el.get())->lexeme.c_str(),
        nullptr, 10));
  }
  return true;
}

// Evaluate an `in::` initializer: a scalar/tensor literal (delegated to
// literal_to_value) OR a registered tensor constructor OpApplication. Returns
// false with a one-line `err` on any malformed constructor.
using InputList = std::vector<std::pair<std::string, interp::Value>>;

// `resolved` holds the inputs already bound, so a property reference can profile
// a matrix declared earlier in the same file. It is null where no such context
// exists (property refs are then rejected with a clear message).
bool eval_initializer(const ast::Expr *e, interp::Value &out, std::string &err,
                      const InputList *resolved) {
  if (!e) { err = "empty initializer"; return false; }

  // Property reference: `m.properties.linear_algebra.determinant`. The profile
  // runs host-side at load time (tier B), and the field is projected into an
  // ordinary integer Value, so downstream flows see a plain tensor.
  if (e->kind == ast::NodeKind::PropertyRef) {
    const auto *pr = static_cast<const ast::PropertyRef *>(e);
    if (!pr->base || pr->base->kind != ast::NodeKind::VarRef) {
      err = "property reference must start from a named input";
      return false;
    }
    const std::string &base = static_cast<const ast::VarRef *>(pr->base.get())->name;
    if (!resolved) {
      err = "property references are not available in this position";
      return false;
    }
    const interp::Value *mat = nullptr;
    for (const auto &kv : *resolved) if (kv.first == base) { mat = &kv.second; break; }
    if (!mat) {
      err = "property base '" + base +
            "' names no input declared above it (declare it with in::" + base + " = ...)";
      return false;
    }
    auto desc = analysis::profile(*mat, base);
    if (!desc.ok) { err = desc.error; return false; }
    std::string perr;
    if (!analysis::descriptorProperty(desc, pr->path, out, perr)) {
      err = perr;
      return false;
    }
    return true;
  }
  if (e->kind == ast::NodeKind::NumberLiteral ||
      e->kind == ast::NodeKind::TensorLiteral) {
    if (literal_to_value(e, out)) return true;
    err = "malformed scalar/tensor literal";
    return false;
  }
  if (e->kind != ast::NodeKind::OpApplication) {
    err = "initializer must be a scalar/tensor literal or a tensor constructor";
    return false;
  }
  const auto *op = static_cast<const ast::OpApplication *>(e);
  const std::string &name = op->op;
  const auto &args = op->args;

  // Tensor constructors share one implementation with the flow-body path
  // (interp/generators.cpp). Here the arguments are literal AST nodes, so they
  // are lowered to Values first and then handed to the same evaluator.
  if (!interp::isGenerator(name)) {
    err = "unknown tensor constructor '" + name + "'";
    return false;
  }
  std::vector<interp::Value> operands;
  for (const auto &a : args) {
    interp::Value av;
    int64_t sc = 0;
    std::vector<int64_t> vec;
    if (init_scalar(a.get(), sc)) {
      av = interp::Value::scalar(sc);
    } else if (init_vector(a.get(), vec)) {
      av = interp::Value::tensor({(int64_t)vec.size()}, vec);
    } else {
      err = name + ": arguments must be integer literals or a literal vector";
      return false;
    }
    operands.push_back(std::move(av));
  }
  std::string gerr;
  auto made = interp::evalGenerator(name, operands, &gerr);
  if (!made) { err = gerr; return false; }
  out = std::move(*made);
  return true;
}

void print_value(const std::string &name, const interp::Value &v) {
  std::cout << name << " : ";
  if (v.is_scalar()) {
    std::cout << "scalar = " << v.data[0] << "\n";
    return;
  }
  std::cout << "[";
  for (size_t i = 0; i < v.dims.size(); ++i) { if (i) std::cout << "x"; std::cout << v.dims[i]; }
  std::cout << "] = [";
  for (size_t i = 0; i < v.data.size(); ++i) { if (i) std::cout << ", "; std::cout << v.data[i]; }
  std::cout << "]\n";
}

// Group an unsigned integer with thousands separators ("1985" -> "1,985").
std::string group_thousands(uint64_t n) {
  std::string s = std::to_string(n);
  for (int pos = static_cast<int>(s.size()) - 3; pos > 0; pos -= 3)
    s.insert(static_cast<size_t>(pos), ",");
  return s;
}

// Human uptime: "2d 11h 01m 26s" (days omitted when zero).
std::string format_uptime(uint64_t secs) {
  uint64_t d = secs / 86400, h = (secs % 86400) / 3600,
           m = (secs % 3600) / 60, s = secs % 60;
  char buf[64];
  if (d)
    std::snprintf(buf, sizeof(buf), "%llud %lluh %02llum %02llus",
                  (unsigned long long)d, (unsigned long long)h,
                  (unsigned long long)m, (unsigned long long)s);
  else
    std::snprintf(buf, sizeof(buf), "%lluh %02llum %02llus",
                  (unsigned long long)h, (unsigned long long)m,
                  (unsigned long long)s);
  return buf;
}

// Render a worker's GET /api/status (PROTOCOL_SPEC.md 6.1) as an aligned
// table. `endpoint`/`state` are not carried by DeviceStatus, so they are
// synthesized from the DeviceConfig and the running/capacity counts.
void print_device_status(const std::string &alias,
                         const net::DeviceConfig &cfg,
                         const net::DeviceStatus &st) {
  const char *rule = "--------------------------------";
  std::string ops;
  for (const auto &op : st.ops) { if (!ops.empty()) ops += ", "; ops += op; }
  char payload[32];
  std::snprintf(payload, sizeof(payload), "%.2f MiB",
                static_cast<double>(st.max_payload) / (1024.0 * 1024.0));
  const bool ready = st.jobs_running < st.jobs_capacity;

  std::printf("Caramel Worker Status\n%s\n", rule);
  // The script's alias is what the user asked about (`status::bob`), so that
  // is the identity to lead with. The worker also self-reports an alias, but
  // it is only informational (net/device.h) and need not match; show it as a
  // separate row when it actually differs rather than silently displaying a
  // name the script never mentions.
  std::printf("%-18s %s\n", "Alias", alias.c_str());
  if (!st.alias.empty() && st.alias != alias)
    std::printf("%-18s %s\n", "Worker name", st.alias.c_str());
  std::printf("%-18s %s\n", "Architecture", st.kind.c_str());
  std::printf("%-18s %d\n", "Protocol", st.proto);
  std::printf("%-18s %s\n", "Uptime", format_uptime(st.uptime_s).c_str());
  std::printf("%-18s %s\n", "Requests", group_thousands(st.requests).c_str());
  std::printf("%-18s %d / %d\n", "Jobs", st.jobs_running, st.jobs_capacity);
  std::printf("%-18s %d\n", "Available slots",
              st.jobs_capacity - st.jobs_running);
  std::printf("%-18s %s\n", "Maximum payload", payload);
  std::printf("%-18s %s\n", "Operations", ops.c_str());
  std::printf("%-18s %s:%u\n", "Endpoint", cfg.host.c_str(),
              static_cast<unsigned>(cfg.port));
  std::printf("%s\n", rule);
  std::printf("%-18s %s\n", "State", ready ? "READY" : "BUSY");
}

// Register the op-set reference evaluators on a local interpreter.
//
// These must match src/net/dispatch.cpp exactly. Without them the interpreter
// has no sigmoid/tanh/softmax/layernorm/conv2d/pooling/reshape/split, so a
// local run either fails with "unsupported op" or — worse — falls through to a
// different rule and quietly returns the wrong values. That is what made
// --verify report MISMATCH against a worker that was answering correctly:
// remote sigmoid gave [500, 731, 881, 119] (the shared fixed-point LUT) while
// the local side produced [1, 1, 1, 0].
//
// quantres matters: activations and layernorm decode register integers at a
// scale of 10^quantres, so the same input yields different (equally correct)
// values at different scales.
// The worker's register file is int32 (bark/kernel/ir_exec.c), so results wrap
// mod 2^32 there while the interpreter computes in int64. They agree only while
// values stay small; a deep chain or a large tensor_sum diverges silently.
void model_worker_register_width(interp::Interpreter &vm) {
  vm.set_op_result_transform([](interp::Value &v) {
    for (auto &e : v.data) { e = static_cast<int32_t>(static_cast<uint32_t>(e)); }
  });
}

void add_op_set_evaluators(interp::Interpreter &vm, int quantres) {
  vm.add_param_evaluator(interp::activationOpEvaluator(quantres));
  vm.add_param_evaluator(interp::spatialOpEvaluator());
  vm.add_param_evaluator(interp::shapeOpEvaluator());
  vm.add_param_evaluator(interp::normOpEvaluator(quantres));
}

// crml::quantres for a local run. Unlike remote execution this is not fatal
// when absent — a flow of pure integer ops does not need a scale — so a missing
// or malformed directive just means 0.
int quantres_or_zero(const ast::Program &program) {
  for (auto &it : program.items) {
    if (it->kind != ast::NodeKind::Directive) continue;
    auto *d = static_cast<ast::Directive *>(it.get());
    if (d->dkind != ast::DirectiveKind::Quant) continue;
    if (d->key != "quantres" && d->key != "decimal_places") continue;
    int64_t v = 0;
    if (parse_i64(d->value, v) && v >= 0 && v <= 255) return static_cast<int>(v);
  }
  return 0;
}

// --- Remote execution (lang_043) -------------------------------------------

// Quant block from the crml::quant* directives (mirrors crpk-gen): the CRPK
// envelope requires them, so remote mode fails locally with a clear message
// when they are absent or do not fit the wire fields.
bool quant_from_directives(const ast::Program &program,
                           proto::QuantBlock &out, std::string &err) {
  int64_t quantmax = 0, quantmin = 0, quantres = 0;
  bool have_max = false, have_min = false, have_res = false;
  for (auto &it : program.items) {
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
    err = "remote execution needs integer crml::quantmax/quantmin/quantres "
          "directives (they fill the CRPK quant block)";
    return false;
  }
  if (quantres < 0 || quantres > 255 ||
      quantmin < std::numeric_limits<int16_t>::min() ||
      quantmin > std::numeric_limits<int16_t>::max() ||
      quantmax < std::numeric_limits<int16_t>::min() ||
      quantmax > std::numeric_limits<int16_t>::max()) {
    err = "quant directives do not fit the CRPK quant block "
          "(quantres in 0..255, quantmin/quantmax in i16)";
    return false;
  }
  out.quantres = static_cast<uint8_t>(quantres);
  out.quantmin = static_cast<int16_t>(quantmin);
  out.quantmax = static_cast<int16_t>(quantmax);
  const auto qp = types::QuantParams::from_range(
      static_cast<double>(quantmax), static_cast<double>(quantmin),
      static_cast<int>(quantres));
  out.dtype = qp.bit_width <= 8 ? proto::kDTypeInt8 : proto::kDTypeInt32;
  return true;
}

// Ship the compiled flow to the worker (pre-flight + POST /api/execute
// ?mode=sync via caramel/net/remote_run.h) and print the decoded outputs in
// the exact local format. With `verify`, also run the local interpreter and
// exact-compare the decoded values (PASS -> 0, MISMATCH -> 1).
int run_remote(const ast::Program &program, const ast::LambdaFlow &flow,
               const ir::DataflowGraph &graph,
               const std::vector<std::pair<std::string, interp::Value>> &inputs,
               const std::string &url, const std::string &user,
               const std::string &pass, const std::string &token,
               bool verify) {
  net::RemoteJob job;
  std::string qerr;
  if (!quant_from_directives(program, job.quant, qerr)) {
    std::cerr << "error: " << qerr << "\n";
    return 1;
  }
  try {
    job.ir = ir::serialize(graph);
  } catch (const std::exception &e) {
    std::cerr << "error: IR serialization failed: " << e.what() << "\n";
    return 1;
  }
  for (const auto &pname : flow.params) {
    const interp::Value *v = nullptr;
    // Last --in wins, matching the local vm.set_input overwrite semantics.
    for (const auto &kv : inputs)
      if (kv.first == pname) v = &kv.second;
    if (!v) {
      std::cerr << "error: no --in value for flow parameter '" << pname << "'\n";
      return 1;
    }
    job.inputs.emplace_back(pname, *v);
  }
  job.output_names = flow.returns;

  auto cfg = net::DeviceConfig::fromCli(url, user, pass, token);
  if (!cfg.ok()) {
    std::cerr << "error: " << cfg.error << "\n";
    return 2;
  }
  net::DeviceSession session(*cfg.config);
  auto remote = net::executeRemoteSync(session, job);
  if (!remote.ok()) {
    std::cerr << "error: " << remote.error << "\n";
    return 1;
  }

  // Identical to the local print path: header line + one line per output in
  // flow return order (RemoteOutputs already carries that order).
  std::cout << "flow '" << flow.name << "' outputs:\n";
  for (const auto &kv : *remote.outputs) print_value(kv.first, kv.second);

  if (verify) {
    interp::Interpreter vm;
    vm.add_evaluator(interp::matrixOpEvaluator());
    vm.add_evaluator(interp::lambdaOpEvaluator());
    add_op_set_evaluators(vm, job.quant.quantres);
    model_worker_register_width(vm);
    for (const auto &kv : inputs) vm.set_input(kv.first, kv.second);
    auto local = vm.run(graph);
    if (!local.ok()) {
      std::cerr << "runtime error: " << local.error->message << "\n";
      return 1;
    }
    const std::string mismatch =
        net::verifyAgainstLocal(*remote.outputs, local.outputs);
    if (!mismatch.empty()) {
      std::cerr << "verify: MISMATCH: " << mismatch << "\n";
      return 1;
    }
    std::cout << "verify: PASS\n";
  }
  return 0;
}

// --- Script-driven dispatch (lang_044) -------------------------------------
// Runs the whole plan (device::-routed flows remotely via net/dispatch.h,
// the rest locally) and prints each flow's outputs in the standard format.
// With `verify`, the plan is re-run all-locally and compared flow by flow.
int run_script_dispatch(
    const ast::Program &program, const std::vector<net::PlannedFlow> &plan,
    const std::vector<net::ScriptDevice> &devices,
    const std::vector<std::pair<std::string, interp::Value>> &inputs,
    bool verify) {
  auto res = net::runFlows(program, plan, devices, inputs);
  if (!res.ok()) {
    std::cerr << "error: " << res.error << "\n";
    return 1;
  }
  for (const auto &fo : res.flows) {
    std::cout << "flow '" << fo.flow_name << "' outputs:\n";
    for (const auto &kv : fo.outputs) print_value(kv.first, kv.second);
  }
  if (verify) {
    std::vector<net::PlannedFlow> local_plan = plan;
    for (auto &p : local_plan) p.device_alias.clear();
    auto local = net::runFlows(program, local_plan, devices, inputs);
    if (!local.ok()) {
      std::cerr << "runtime error: " << local.error << "\n";
      return 1;
    }
    for (size_t i = 0; i < res.flows.size(); ++i) {
      std::unordered_map<std::string, interp::Value> lmap;
      for (const auto &kv : local.flows[i].outputs)
        lmap.emplace(kv.first, kv.second);
      const std::string mismatch =
          net::verifyAgainstLocal(res.flows[i].outputs, lmap);
      if (!mismatch.empty()) {
        std::cerr << "verify: MISMATCH in flow '" << res.flows[i].flow_name
                  << "': " << mismatch << "\n";
        return 1;
      }
    }
    std::cout << "verify: PASS\n";
  }
  return 0;
}

// lang_045: `caramel-run devices` - probe the LAN (PROTOCOL_SPEC.md
// Section 4) and print one line per worker. Exit 0 even when nobody
// answers; exit 2 on bad flags, 1 on a socket-level failure.
int run_devices_command(int argc, char **argv) {
  net::DiscoveryOptions opts;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--probe" && i + 1 < argc) { opts.probes.push_back(argv[++i]); }
    else if (a == "--timeout" && i + 1 < argc) {
      opts.timeout_ms = std::atoi(argv[++i]);
      if (opts.timeout_ms <= 0) { std::cerr << "bad --timeout\n"; return 2; }
    }
    else if (a == "--no-broadcast") { opts.broadcast = false; }
    else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
  }
  auto res = net::discoverDevices(opts);
  if (!res.ok()) { std::cerr << "error: " << res.error << "\n"; return 1; }
  if (res.devices.empty()) { std::cout << "no devices found\n"; return 0; }
  std::printf("%-18s %-5s %-21s %-5s %s\n",
              "ALIAS", "KIND", "ADDRESS", "BUSY", "OPS");
  for (const auto &d : res.devices) {
    std::string ops;
    for (const auto &op : d.ops) { if (!ops.empty()) ops += ","; ops += op; }
    std::printf("%-18s %-5s %-21s %-5s %s\n", d.alias.c_str(),
                d.kind.c_str(), d.connectHost().c_str(),
                d.busy ? "yes" : "no", ops.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: caramel-run <file.crml> [--flow NAME] [--in name=VALUE]...\n"
                 "                   [--device http://HOST:PORT] [--device-user USER]\n"
                 "                   [--device-pass PASS] [--device-token TOK] [--verify]\n"
                 "                   [--probe HOST[:PORT]]... [--discover-timeout MS]\n"
                 "       caramel-run devices [--probe HOST[:PORT]]... [--timeout MS]\n"
                 "                           [--no-broadcast]\n";
    return 2;
  }
  if (std::string(argv[1]) == "devices") {
    return run_devices_command(argc, argv);
  }
  std::string path = argv[1];
  std::string flow_name;
  std::string device_url, device_user, device_pass, device_token;
  bool verify = false;
  // --profile-json: emit profile:: results as canonical JSON instead of the
  // human report, so a profile can be diffed or fed to another tool.
  bool profile_json = false;
  net::DiscoveryOptions discover_opts;  // lang_045: for discover = true blocks
  std::vector<std::pair<std::string, interp::Value>> inputs;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--flow" && i + 1 < argc) { flow_name = argv[++i]; }
    else if (a == "--in" && i + 1 < argc) {
      std::string n; interp::Value v;
      if (!parse_input(argv[++i], n, v)) { std::cerr << "bad --in: " << argv[i] << "\n"; return 2; }
      inputs.emplace_back(std::move(n), std::move(v));
    }
    else if (a == "--device" && i + 1 < argc) { device_url = argv[++i]; }
    else if (a == "--device-user" && i + 1 < argc) { device_user = argv[++i]; }
    else if (a == "--device-pass" && i + 1 < argc) { device_pass = argv[++i]; }
    else if (a == "--device-token" && i + 1 < argc) { device_token = argv[++i]; }
    else if (a == "--verify") { verify = true; }
    else if (a == "--profile-json") { profile_json = true; }
    else if (a == "--probe" && i + 1 < argc) { discover_opts.probes.push_back(argv[++i]); }
    else if (a == "--discover-timeout" && i + 1 < argc) {
      discover_opts.timeout_ms = std::atoi(argv[++i]);
      if (discover_opts.timeout_ms <= 0) { std::cerr << "bad --discover-timeout\n"; return 2; }
    }
    else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
  }
  // lang_044: --verify is now also legal without --device when the script
  // itself routes flows with @device (checked after parsing); the credential
  // flags still only make sense with --device.
  if (device_url.empty() &&
      (!device_user.empty() || !device_pass.empty() || !device_token.empty())) {
    std::cerr << "error: --device-user/--device-pass/--device-token "
                 "require --device\n";
    return 2;
  }

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
  // lang_044: non-fatal diagnostics (e.g. a plaintext pass in a device block).
  for (const auto &w : parser.warnings())
    std::cerr << "warning [" << w.loc.line << ":" << w.loc.column << "] "
              << w.message << "\n";

  // status::alias; host-side queries: print a declared worker's
  // GET /api/status metadata (PROTOCOL_SPEC.md 6.1). Not a dataflow op, so it
  // runs here in the driver, before flow execution.
  bool handled_status = false;
  {
    bool any_status = false;
    for (auto &it : program->items) {
      if (it->kind != ast::NodeKind::Directive) continue;
      if (static_cast<ast::Directive *>(it.get())->dkind ==
          ast::DirectiveKind::Status) { any_status = true; break; }
    }
    if (any_status) {
      std::vector<net::ScriptDevice> devices;
      std::string derr;
      if (!net::collectScriptDevices(*program, devices, derr, &discover_opts)) {
        std::cerr << "error: " << derr << "\n";
        return 1;
      }
      for (auto &it : program->items) {
        if (it->kind != ast::NodeKind::Directive) continue;
        auto *d = static_cast<ast::Directive *>(it.get());
        if (d->dkind != ast::DirectiveKind::Status) continue;
        const net::ScriptDevice *dev = nullptr;
        for (const auto &sd : devices) if (sd.alias == d->key) { dev = &sd; break; }
        if (!dev) {
          std::cerr << "error: status::" << d->key
                    << " names no device:: block\n";
          return 1;
        }
        net::DeviceSession session(dev->config);
        auto st = session.status();
        if (!st.ok()) {
          std::cerr << "error: status::" << d->key << ": " << st.error.message
                    << "\n";
          return 1;
        }
        print_device_status(d->key, dev->config, *st.status);
      }
      handled_status = true;
    }
  }

  // Script-embedded inputs: `in::name = <literal>;` fills a flow parameter so
  // no --in is needed (a self-contained script). A matching --in overrides.
  for (auto &it : program->items) {
    if (it->kind != ast::NodeKind::Directive) continue;
    auto *d = static_cast<ast::Directive *>(it.get());
    if (d->dkind != ast::DirectiveKind::Input || !d->valueExpr) continue;
    bool overridden = false;
    for (const auto &kv : inputs) if (kv.first == d->key) { overridden = true; break; }
    if (overridden) continue;
    interp::Value v;
    std::string init_err;
    if (!eval_initializer(d->valueExpr.get(), v, init_err, &inputs)) {
      std::cerr << "error: in::" << d->key << ": " << init_err << "\n";
      return 1;
    }
    inputs.emplace_back(d->key, std::move(v));
  }

  // profile::name; host-side algebraic profile of an input matrix. Runs after
  // inputs are resolved so it can read `in::` matrices (and --in overrides).
  bool handled_profile = false;
  for (auto &it : program->items) {
    if (it->kind != ast::NodeKind::Directive) continue;
    auto *d = static_cast<ast::Directive *>(it.get());
    if (d->dkind != ast::DirectiveKind::Profile) continue;
    const interp::Value *mat = nullptr;
    for (const auto &kv : inputs) if (kv.first == d->key) { mat = &kv.second; break; }
    if (!mat) {
      std::cerr << "error: profile::" << d->key
                << " names no input (declare it with in::" << d->key << " = ...)\n";
      return 1;
    }
    auto desc = analysis::profile(*mat, d->key);
    if (!desc.ok) {
      std::cerr << "error: profile::" << d->key << ": " << desc.error << "\n";
      return 1;
    }
    std::cout << (profile_json ? analysis::renderJson(desc) + "\n"
                               : analysis::renderReport(desc));
    handled_profile = true;
  }

  // lang_044: script-driven routing. Precedence: an explicit --device flag
  // WINS over in-script @device routing (the operator's command line
  // overrides the script, and the lang_043 single-flow sync path stays
  // untouched). Script mode activates when --device is absent and at least
  // one selected flow carries @device(alias); it runs ALL selected flows
  // (every lambda_flow, or just --flow's) with dependency-ordered,
  // concurrency-exploiting dispatch.
  if (device_url.empty()) {
    std::vector<net::PlannedFlow> plan;
    bool any_routed = false;
    for (auto &it : program->items) {
      if (it->kind != ast::NodeKind::LambdaFlow) continue;
      auto *lf = static_cast<ast::LambdaFlow *>(it.get());
      if (!flow_name.empty() && lf->name != flow_name) continue;
      net::PlannedFlow pf;
      pf.flow = lf;
      pf.device_alias = net::deviceAliasOf(*lf);
      if (!pf.device_alias.empty()) any_routed = true;
      plan.push_back(pf);
    }
    if (any_routed) {
      std::vector<net::ScriptDevice> devices;
      std::string derr;
      if (!net::collectScriptDevices(*program, devices, derr,
                                     &discover_opts)) {
        std::cerr << "error: " << derr << "\n";
        return 1;
      }
      return run_script_dispatch(*program, plan, devices, inputs, verify);
    }
    if (verify) {
      std::cerr << "error: --verify requires --device or a script with "
                   "@device routing\n";
      return 2;
    }
  }

  const ast::LambdaFlow *flow = nullptr;
  for (auto &it : program->items)
    if (it->kind == ast::NodeKind::LambdaFlow) {
      auto *lf = static_cast<ast::LambdaFlow *>(it.get());
      if (flow_name.empty() || lf->name == flow_name) { flow = lf; break; }
    }
  if (!flow) {
    // A script may legitimately contain only status::/profile:: actions.
    if ((handled_status || handled_profile) && flow_name.empty()) return 0;
    std::cerr << "error: no lambda_flow"
              << (flow_name.empty() ? "" : " named '" + flow_name + "'")
              << " found\n";
    return 1;
  }

  auto graph = ir::buildDataflow(*flow);

  // Remote mode (lang_043) is strictly behind --device: the default local
  // path below is untouched.
  if (!device_url.empty()) {
    return run_remote(*program, *flow, graph, inputs, device_url, device_user,
                      device_pass, device_token, verify);
  }

  interp::Interpreter vm;
  vm.add_evaluator(interp::matrixOpEvaluator());
  vm.add_evaluator(interp::lambdaOpEvaluator());
  add_op_set_evaluators(vm, quantres_or_zero(*program));
  model_worker_register_width(vm);
  for (auto &kv : inputs) vm.set_input(kv.first, kv.second);

  auto result = vm.run(graph);
  if (!result.ok()) {
    std::cerr << "runtime error: " << result.error->message << "\n";
    return 1;
  }

  std::cout << "flow '" << flow->name << "' outputs:\n";
  for (const auto &name : flow->returns) {
    auto it = result.outputs.find(name);
    if (it != result.outputs.end()) print_value(name, it->second);
  }
  return 0;
}
