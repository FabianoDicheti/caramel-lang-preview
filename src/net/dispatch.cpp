// ============================================================================
// Caramel Language - Concurrent flow dispatch implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_044 (Async Jobs Client + device:: Syntax in .crml)
// Version: 1.0.0
// ----------------------------------------------------------------------------
// See include/caramel/net/dispatch.h for the behavioral contract.
// ============================================================================
#include "caramel/net/dispatch.h"

#include <cstdlib>
#include <exception>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "caramel/interp/activation.h"
#include "caramel/interp/interpreter.h"
#include "caramel/interp/lambda.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/interp/norm.h"
#include "caramel/interp/shape.h"
#include "caramel/interp/spatial.h"
#include "caramel/ir/binary_ir.h"
#include "caramel/ir/dataflow.h"
#include "caramel/types/type.h"

namespace caramel::net {
namespace {

namespace ast = caramel::ast;
namespace interp = caramel::interp;
namespace ir = caramel::ir;
namespace proto = caramel::proto;

bool parseI64(const std::string &s, int64_t &out) {
  if (s.empty()) return false;
  size_t pos = 0;
  try {
    out = std::stoll(s, &pos);
  } catch (const std::exception &) {
    return false;
  }
  return pos == s.size();
}

// Resolve a non-secret device field: env("NAME") fields read the variable
// here (empty string when unset); literals pass through.
std::string plainField(const ast::DeviceField &f) {
  if (!f.is_env) return f.value;
  const char *v = std::getenv(f.value.c_str());
  return v ? std::string(v) : std::string();
}

// Resolve a SECRET device field into the lang_042 "env:NAME" spelling so
// the DeviceConfig/DeviceSession machinery reads the variable at run time
// and the literal never appears in any diagnostic.
std::string secretField(const ast::DeviceField &f) {
  if (!f.is_env) return f.value;
  return "env:" + f.value;
}

// Run one flow on the local interpreter; outputs land in `fo.outputs` in
// flow return order. Returns "" on success.
std::string runLocalFlow(
    const ast::LambdaFlow &flow, int quantres,
    const std::vector<std::pair<std::string, interp::Value>> &inputs,
    FlowOutputs &fo) {
  ir::DataflowGraph graph = ir::buildDataflow(flow);
  interp::Interpreter vm;
  // The worker's register file is int32 (bark/kernel/ir_exec.c: rf_slot_t holds
  // int32_t), so every op result there wraps mod 2^32. The interpreter computes
  // in int64, which agrees only while values stay small — a deep matmul chain or
  // a tensor_sum over a large tensor diverges silently. Model the same register
  // width so --verify compares like with like.
  //
  // Wrapping (not saturating) is the correct model and is load-bearing: sum mod
  // 2^32 is an abelian group, which is exactly why the worker's tiled and
  // sliced matmul is bit-exact under reordering.
  vm.set_op_result_transform([](interp::Value &v) {
    for (auto &e : v.data) { e = static_cast<int32_t>(static_cast<uint32_t>(e)); }
  });
  vm.add_evaluator(interp::matrixOpEvaluator());
  vm.add_evaluator(interp::lambdaOpEvaluator());
  // Activations need the decimal fixed-point scale (10^quantres) to interpret
  // register integers; the shared LUT kernel keeps this byte-identical to the
  // worker so --verify passes.
  vm.add_param_evaluator(interp::activationOpEvaluator(quantres));
  vm.add_param_evaluator(interp::spatialOpEvaluator());
  vm.add_param_evaluator(interp::shapeOpEvaluator());
  vm.add_param_evaluator(interp::normOpEvaluator(quantres));
  for (const auto &kv : inputs) vm.set_input(kv.first, kv.second);
  auto result = vm.run(graph);
  if (!result.ok()) {
    return "runtime error: " + result.error->message;
  }
  for (const auto &name : flow.returns) {
    auto it = result.outputs.find(name);
    if (it == result.outputs.end()) {
      return "local run produced no output named '" + name + "'";
    }
    fo.outputs.emplace_back(name, it->second);
  }
  return std::string();
}

// Run one flow remotely through executeRemoteAuto. Returns "" on success.
std::string runRemoteFlow(
    const ast::LambdaFlow &flow, const proto::QuantBlock &quant,
    const DeviceConfig &cfg,
    const std::vector<std::pair<std::string, interp::Value>> &inputs,
    const JobsOptions &jobs, FlowOutputs &fo) {
  RemoteJob job;
  job.quant = quant;
  try {
    ir::DataflowGraph graph = ir::buildDataflow(flow);
    job.ir = ir::serialize(graph);
  } catch (const std::exception &e) {
    return std::string("IR serialization failed: ") + e.what();
  }
  job.inputs = inputs;
  job.output_names = flow.returns;

  // Fresh session per flow: DeviceSession is not thread-safe and flows in
  // one level run on distinct threads.
  DeviceSession session(cfg);
  auto res = executeRemoteAuto(session, job, jobs);
  if (!res.ok()) return res.error;
  fo.outputs = std::move(*res.outputs);
  return std::string();
}

}  // namespace

bool collectScriptDevices(const ast::Program &program,
                          std::vector<ScriptDevice> &out, std::string &err,
                          const DiscoveryOptions *discovery) {
  out.clear();
  // lang_045: one lazy probe sweep shared by every `discover = true` block.
  bool probed = false;
  DiscoveryResult found;
  for (const auto &it : program.items) {
    if (it->kind != ast::NodeKind::DeviceBlock) continue;
    const auto *db = static_cast<const ast::DeviceBlock *>(it.get());
    if (db->device != ast::Device::Remote) continue;  // legacy fpga/cpu form
    for (const auto &d : out) {
      if (d.alias == db->alias) {
        err = "duplicate device block: device::" + db->alias +
              " is defined more than once";
        return false;
      }
    }
    std::string host, port, user, pass, token, discover;
    for (const auto &f : db->fields) {
      if (f.key == "host") host = plainField(f);
      else if (f.key == "port") port = plainField(f);
      else if (f.key == "user") user = plainField(f);
      else if (f.key == "pass") pass = secretField(f);
      else if (f.key == "token") token = secretField(f);
      else if (f.key == "discover") discover = plainField(f);
      // unknown keys (e.g. a future `kind`): ignored, forward compatible
    }
    const bool want_discovery = (discover == "true" || discover == "1");
    if (!discover.empty() && !want_discovery && discover != "false" &&
        discover != "0") {
      err = "device::" + db->alias + ": discover must be true or false, got '" +
            discover + "'";
      return false;
    }
    if (want_discovery && !host.empty()) {
      err = "device::" + db->alias +
            ": host and discover = true are mutually exclusive";
      return false;
    }
    if (want_discovery) {
      if (!port.empty()) {
        err = "device::" + db->alias +
              ": port comes from the discovery reply; drop the port field";
        return false;
      }
      if (!probed) {
        found = discoverDevices(discovery ? *discovery : DiscoveryOptions{});
        probed = true;
        if (!found.ok()) {
          err = "device::" + db->alias + ": " + found.error;
          return false;
        }
      }
      const DiscoveredDevice *match = nullptr;
      for (const auto &dev : found.devices) {
        if (dev.alias == db->alias) { match = &dev; break; }
      }
      if (!match) {
        err = "device::" + db->alias + ": no worker with alias '" + db->alias +
              "' answered discovery (" + std::to_string(found.devices.size()) +
              (found.devices.size() == 1 ? " reply" : " replies") + " received)";
        return false;
      }
      host = match->connectHost();
    }
    if (host.empty()) {
      err = "device::" + db->alias + " has no host field (or discover = true)";
      return false;
    }
    std::string url = host;
    if (!port.empty() && host.find(':') == std::string::npos &&
        host.rfind("http://", 0) != 0) {
      int64_t p = 0;
      if (!parseI64(port, p) || p <= 0 || p > 65535) {
        err = "device::" + db->alias + " has an invalid port '" + port + "'";
        return false;
      }
      url += ":" + port;
    }
    auto cfg = DeviceConfig::fromCli(url, user, pass, token);
    if (!cfg.ok()) {
      err = "device::" + db->alias + ": " + cfg.error;
      return false;
    }
    cfg.config->alias = db->alias;
    out.push_back(ScriptDevice{db->alias, std::move(*cfg.config)});
  }
  return true;
}

std::string deviceAliasOf(const ast::LambdaFlow &flow) {
  for (const auto &dec : flow.decorators) {
    if (dec.name != "device" || !dec.args) continue;
    for (const auto &f : dec.args->fields) {
      if (!f.key.empty() && f.key != "target") continue;
      if (!f.value || f.value->kind != ast::NodeKind::VarRef) continue;
      const auto &name = static_cast<const ast::VarRef &>(*f.value).name;
      if (name == "fpga" || name == "cpu") return std::string();  // legacy
      return name;
    }
  }
  return std::string();
}

bool quantFromProgram(const ast::Program &program, proto::QuantBlock &out,
                      std::string &err) {
  int64_t quantmax = 0, quantmin = 0, quantres = 0;
  bool have_max = false, have_min = false, have_res = false;
  for (const auto &it : program.items) {
    if (it->kind != ast::NodeKind::Directive) continue;
    const auto *d = static_cast<const ast::Directive *>(it.get());
    if (d->dkind != ast::DirectiveKind::Quant) continue;
    int64_t v = 0;
    if (!parseI64(d->value, v)) continue;
    if (d->key == "quantmax") { quantmax = v; have_max = true; }
    if (d->key == "quantmin") { quantmin = v; have_min = true; }
    if (d->key == "quantres" || d->key == "decimal_places") {
      quantres = v;
      have_res = true;
    }
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
  const auto qp = caramel::types::QuantParams::from_range(
      static_cast<double>(quantmax), static_cast<double>(quantmin),
      static_cast<int>(quantres));
  out.dtype = qp.bit_width <= 8 ? proto::kDTypeInt8 : proto::kDTypeInt32;
  return true;
}

DispatchResult runFlows(
    const ast::Program &program, const std::vector<PlannedFlow> &plan,
    const std::vector<ScriptDevice> &devices,
    const std::vector<std::pair<std::string, interp::Value>> &cli_inputs,
    const JobsOptions &jobs) {
  DispatchResult out;
  const size_t n = plan.size();
  if (n == 0) return out;

  // --- Resolve device aliases up front ------------------------------------
  std::vector<const DeviceConfig *> cfg_of(n, nullptr);
  bool any_remote = false;
  for (size_t i = 0; i < n; ++i) {
    if (!plan[i].flow) {
      out.error = "internal: execution plan entry " + std::to_string(i) +
                  " has no flow";
      return out;
    }
    if (plan[i].device_alias.empty()) continue;
    for (const auto &d : devices) {
      if (d.alias == plan[i].device_alias) {
        cfg_of[i] = &d.config;
        break;
      }
    }
    if (!cfg_of[i]) {
      out.error = "flow '" + plan[i].flow->name + "' targets @device(" +
                  plan[i].device_alias + ") but no device::" +
                  plan[i].device_alias + " block defines it";
      return out;
    }
    any_remote = true;
  }

  // Quant block: mandatory once any flow goes remote (it fills the CRPK
  // envelope), but parsed for local-only runs too. quantres is the decimal
  // fixed-point scale that activations and layernorm decode register integers
  // with, so skipping it locally silently evaluates them at scale 1 — which is
  // what made `--verify` report MISMATCH against a worker that was answering
  // correctly: the verify pass re-runs the plan with every device alias
  // cleared, so any_remote was false and sigmoid came back [1,1,1,0] instead of
  // [500,731,881,119]. Absent directives stay non-fatal for a local run.
  proto::QuantBlock quant;
  {
    std::string qerr;
    if (!quantFromProgram(program, quant, qerr) && any_remote) {
      out.error = qerr;
      return out;
    }
  }

  // --- Dependency graph: B depends on A when a param of B is a return of A.
  std::unordered_map<std::string, size_t> producer;  // output name -> flow
  for (size_t i = 0; i < n; ++i) {
    for (const auto &ret : plan[i].flow->returns) {
      auto ins = producer.emplace(ret, i);
      if (!ins.second && ins.first->second != i) {
        out.error = "flows '" + plan[ins.first->second].flow->name +
                    "' and '" + plan[i].flow->name +
                    "' both return '" + ret +
                    "'; producers must be unambiguous for dispatch";
        return out;
      }
    }
  }
  std::vector<std::unordered_set<size_t>> deps(n);
  for (size_t i = 0; i < n; ++i) {
    for (const auto &p : plan[i].flow->params) {
      auto it = producer.find(p);
      if (it != producer.end() && it->second != i) deps[i].insert(it->second);
    }
  }

  // Kahn layering into dependency levels (cycle -> error).
  std::vector<int> level(n, -1);
  size_t assigned = 0;
  std::vector<std::vector<size_t>> levels;
  while (assigned < n) {
    std::vector<size_t> ready;
    for (size_t i = 0; i < n; ++i) {
      if (level[i] >= 0) continue;
      bool ok = true;
      for (size_t d : deps[i]) {
        if (level[d] < 0) { ok = false; break; }
      }
      if (ok) ready.push_back(i);
    }
    if (ready.empty()) {
      std::string names;
      for (size_t i = 0; i < n; ++i) {
        if (level[i] >= 0) continue;
        if (!names.empty()) names += ", ";
        names += "'" + plan[i].flow->name + "'";
      }
      out.error = "circular dependency between flows " + names +
                  " (each consumes another's output); break the cycle";
      return out;
    }
    for (size_t i : ready) level[i] = static_cast<int>(levels.size());
    assigned += ready.size();
    levels.push_back(std::move(ready));
  }

  // --- Execute level by level ---------------------------------------------
  std::unordered_map<std::string, interp::Value> produced;
  std::vector<FlowOutputs> results(n);
  std::vector<std::string> errors(n);

  for (const auto &lvl : levels) {
    // Resolve inputs on the coordinating thread (reads `produced`).
    std::vector<std::vector<std::pair<std::string, interp::Value>>> ins(
        lvl.size());
    for (size_t k = 0; k < lvl.size(); ++k) {
      const size_t i = lvl[k];
      for (const auto &p : plan[i].flow->params) {
        auto it = produced.find(p);
        if (it != produced.end()) {
          ins[k].emplace_back(p, it->second);
          continue;
        }
        const interp::Value *v = nullptr;
        for (const auto &kv : cli_inputs) {
          if (kv.first == p) v = &kv.second;  // last one wins (--in semantics)
        }
        if (!v) {
          out.error = "flow '" + plan[i].flow->name + "' parameter '" + p +
                      "' has no value: pass --in " + p +
                      "=... or produce it from an earlier flow";
          return out;
        }
        ins[k].emplace_back(p, *v);
      }
    }

    // One thread per flow in the level (a singleton level runs inline).
    auto runOne = [&](size_t k) {
      const size_t i = lvl[k];
      results[i].flow_name = plan[i].flow->name;
      results[i].device_alias = plan[i].device_alias;
      if (cfg_of[i]) {
        errors[i] = runRemoteFlow(*plan[i].flow, quant, *cfg_of[i], ins[k],
                                  jobs, results[i]);
      } else {
        errors[i] = runLocalFlow(*plan[i].flow, quant.quantres, ins[k], results[i]);
      }
    };
    if (lvl.size() == 1) {
      runOne(0);
    } else {
      std::vector<std::thread> threads;
      threads.reserve(lvl.size());
      for (size_t k = 0; k < lvl.size(); ++k)
        threads.emplace_back(runOne, k);
      for (auto &t : threads) t.join();
    }

    // First failing flow in plan order wins the report.
    for (size_t i = 0; i < n; ++i) {
      if (!errors[i].empty()) {
        out.error = "flow '" + plan[i].flow->name + "': " + errors[i];
        return out;
      }
    }
    // Merge produced tensors for the next level (single-threaded).
    for (size_t k = 0; k < lvl.size(); ++k) {
      const size_t i = lvl[k];
      for (const auto &kv : results[i].outputs) {
        produced[kv.first] = kv.second;
      }
    }
  }

  out.flows.reserve(n);
  for (size_t i = 0; i < n; ++i) out.flows.push_back(std::move(results[i]));
  return out;
}

}  // namespace caramel::net
