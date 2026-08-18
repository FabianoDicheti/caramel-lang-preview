// ============================================================================
// Caramel Language - Concurrent flow dispatch (script-driven remote routing)
// ----------------------------------------------------------------------------
// Ticket:   lang_044 (Async Jobs Client + device:: Syntax in .crml)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// The controller behind script-driven routing: a .crml script declares its
// workers with `device::alias { host = ...; user = ...; pass = env("N"); }`
// blocks and routes flows with `@device(alias)`; this module turns that into
// a dependency-ordered, concurrency-exploiting execution plan.
//
// Scheduling model (documented design decision): LEVEL-SYNCHRONOUS
// THREAD-PER-FLOW. Flow B depends on flow A when a parameter of B is named
// among A's returns; Kahn layering groups flows into dependency levels.
// Every flow in a level runs on its own std::thread (levels of size one run
// inline), so independent flows targeting different devices overlap their
// in-flight requests; the level joins before the next level starts, and the
// controller moves produced tensors between devices (workers never talk to
// each other). Each flow gets a FRESH DeviceSession - sessions are
// explicitly not thread-safe (net/device.h) and CDP tokens never expire, so
// per-flow logins are cheap and safe.
//
// Input resolution per flow parameter: a tensor produced by an earlier
// level wins over a caller-supplied (--in) input of the same name; a
// parameter with neither is a hard, actionable error.
//
// Transport per remote flow is executeRemoteAuto (net/jobs.h): sync under
// the 64 KiB threshold, async + polling above it. Local flows (no @device,
// or the legacy fpga/cpu routing decorator) run on the in-process
// interpreter, byte-identical to a plain local run.
//
// Error handling follows the repo idiom: no exceptions cross the API;
// DispatchResult carries either per-flow outputs (plan order) or a one-line
// error message ready for stderr, prefixed with the failing flow. Secrets
// never appear in any message.
// ============================================================================
#ifndef CARAMEL_NET_DISPATCH_H
#define CARAMEL_NET_DISPATCH_H

#include <string>
#include <utility>
#include <vector>

#include "caramel/ast/ast.h"
#include "caramel/interp/value.h"
#include "caramel/net/device.h"
#include "caramel/net/discovery.h"
#include "caramel/net/jobs.h"
#include "caramel/proto/crpk.h"

namespace caramel::net {

// One worker declared by a device::alias block, resolved to a lang_042
// DeviceConfig. env("NAME") secrets are handed to DeviceConfig::fromCli in
// the "env:NAME" spelling, which reads the variable at collection (run)
// time - never at parse time - and never echoes it in any diagnostic.
struct ScriptDevice {
  std::string alias;
  DeviceConfig config;
};

// Collect every device::alias block of the program. Recognized fields:
// host, port, user, pass, token, discover; unknown keys (e.g. a future
// `kind`) are ignored for forward compatibility. Exactly one of `host` or
// `discover = true` is required per block. Returns false + a one-line `err`
// on a duplicate alias or an unusable configuration.
//
// lang_045: `discover = true;` resolves the block's address at collection
// time by probing the LAN (PROTOCOL_SPEC.md Section 4) and binding to the
// reply whose advertised alias equals the block name; the connect address is
// the reply's source IP + advertised port (see net/discovery.h). Discovery
// runs at most ONCE per call, lazily, shared by every discover block.
// `discovery` overrides the probe options (nullptr = defaults: broadcast,
// 1 s window); a block whose alias nobody claims is a hard error naming it.
bool collectScriptDevices(const caramel::ast::Program &program,
                          std::vector<ScriptDevice> &out, std::string &err,
                          const DiscoveryOptions *discovery = nullptr);

// The routing alias of a flow: the first @device decorator argument (bare
// `@device(bob)` or keyed `@device{target: bob}`). Returns "" for local
// execution - no @device, argument-less @device, or the legacy hardware
// targets "fpga"/"cpu" (which lang_044 leaves to the local pipeline).
std::string deviceAliasOf(const caramel::ast::LambdaFlow &flow);

// Quant block from the crml::quant* directives (library form of the
// caramel-run helper; the CRPK envelope requires it for any remote flow).
bool quantFromProgram(const caramel::ast::Program &program,
                      caramel::proto::QuantBlock &out, std::string &err);

// One flow of the execution plan. `device_alias` == "" runs locally.
struct PlannedFlow {
  const caramel::ast::LambdaFlow *flow = nullptr;
  std::string device_alias;
};

// Decoded outputs of one executed flow, in flow return order.
struct FlowOutputs {
  std::string flow_name;
  std::string device_alias;  // "" = ran locally
  RemoteOutputs outputs;
};

// Expected-style result: per-flow outputs in plan order, or a one-line
// error. On error, `flows` holds whatever completed before the failure.
struct DispatchResult {
  std::vector<FlowOutputs> flows;
  std::string error;

  bool ok() const { return error.empty(); }
};

// Execute the plan per the header comment. `devices` comes from
// collectScriptDevices; `cli_inputs` are caller-supplied name/value pairs
// (last one wins, matching --in semantics); `jobs` tunes the async client
// (tests inject a recording sleep hook).
DispatchResult runFlows(
    const caramel::ast::Program &program, const std::vector<PlannedFlow> &plan,
    const std::vector<ScriptDevice> &devices,
    const std::vector<std::pair<std::string, caramel::interp::Value>>
        &cli_inputs,
    const JobsOptions &jobs = JobsOptions{});

}  // namespace caramel::net

#endif  // CARAMEL_NET_DISPATCH_H
