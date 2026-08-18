// ============================================================================
// Caramel Language - Remote flow execution (CDP client side, sync mode)
// ----------------------------------------------------------------------------
// Ticket:   lang_043 (Remote Execution in caramel-run)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// The controller pipeline behind `caramel-run --device`: package a compiled
// flow (binary IR + quant block + input tensors) as a CRPK envelope
// (lang_040), pre-flight it against GET /api/status, submit it via
// POST /api/execute?mode=sync through an authenticated DeviceSession
// (lang_042), and decode the CRRS response back into interpreter values.
//
// Pre-flight (PROTOCOL_SPEC.md Sections 3, 6.1, 6.5) - all three checks run
// BEFORE /api/execute is ever contacted, each failing locally:
//   * proto == 1                       (else PROTO_VERSION)
//   * every compute opcode mnemonic used by the flow IR is in the worker's
//     advertised `ops` list           (else UNSUPPORTED_OP, naming the
//                                      missing mnemonics - no wasted trip)
//   * CRPK size <= max_payload        (else TOO_LARGE)
// Structural opcodes (NOP/PARAM/CONST/INPUT/SYNC/RETURN) are inherent to
// executing any IR module and are not capability-gated by `ops` (the spec's
// Section 6.1 example list carries compute mnemonics only).
//
// Error mapping: every Section 7 error code maps to one distinct, actionable,
// one-line CLI message via remoteErrorMessage(); worker-supplied `detail` is
// appended in brackets. AUTH_* codes surfaced by DeviceSession are routed
// through the same mapping so the CLI never shows two spellings.
//
// Error handling follows the repo idiom (net/device.h): no exceptions cross
// the API; RemoteRunResult carries either the decoded outputs or a one-line
// error message ready for stderr. Secrets never appear in any message.
// ============================================================================
#ifndef CARAMEL_NET_REMOTE_RUN_H
#define CARAMEL_NET_REMOTE_RUN_H

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "caramel/interp/value.h"
#include "caramel/ir/binary_ir.h"
#include "caramel/net/device.h"
#include "caramel/proto/crpk.h"

namespace caramel::net {

// One compiled flow invocation, ready for remote dispatch. `inputs` are in
// IR param-slot order (flow parameter declaration order); `output_names` in
// flow return order (= CRRS output-table order).
struct RemoteJob {
  proto::QuantBlock quant;
  std::vector<uint8_t> ir;  // verbatim caramel::ir::serialize() output
  std::vector<std::pair<std::string, caramel::interp::Value>> inputs;
  std::vector<std::string> output_names;
};

// Named, decoded output tensors in flow return order.
using RemoteOutputs =
    std::vector<std::pair<std::string, caramel::interp::Value>>;

// Expected-style result: exactly one of `outputs` or a nonempty one-line
// `error` message (ready for the CLI, never contains secrets).
struct RemoteRunResult {
  std::optional<RemoteOutputs> outputs;
  std::string error;

  bool ok() const { return outputs.has_value(); }
};

// Canonical worker `ops` mnemonic for a compute opcode (PROTOCOL_SPEC.md
// Sections 4/6.1 spelling, e.g. MATMUL -> "matmul"). nullptr for structural
// opcodes (NOP/PARAM/CONST/INPUT/SYNC/RETURN) and UNKNOWN.
const char *workerMnemonic(caramel::ir::OpCode op);

// Distinct compute mnemonics used by a serialized IR module, in first-use
// order. nullopt if the bytes are not a valid IR module.
std::optional<std::vector<std::string>> requiredOps(
    const std::vector<uint8_t> &ir);

// One distinct, actionable, one-line CLI message per Section 7 error code
// (unknown codes get a generic line quoting the code). `detail` is the
// worker-supplied diagnostic, appended when nonempty.
std::string remoteErrorMessage(const std::string &code,
                               const std::string &detail);

// ---------------------------------------------------------------------------
// lang_044: shared pipeline stages. These are the exact stages the lang_043
// sync path is built from, exposed so the async jobs client (net/jobs.h)
// reuses them; every message is byte-identical to lang_043 behavior.
// ---------------------------------------------------------------------------

// "host:port" label used in every worker-referencing error message.
std::string deviceLabel(const DeviceSession &session);

// Map a DeviceSession failure to a CLI line: Section 7 codes go through
// remoteErrorMessage(); transport/config/protocol failures keep the
// session's own message (they have no wire code).
std::string sessionErrorText(const DeviceError &err);

// Encode a job as a CRPK envelope and compute the distinct compute
// mnemonics it needs. Returns "" on success (filling `crpk` and
// `ops_needed`), else a one-line error message.
std::string encodeJobCrpk(const RemoteJob &job, std::vector<uint8_t> &crpk,
                          std::vector<std::string> &ops_needed);

// The three GET /api/status pre-flight checks (proto==1, ops coverage,
// size <= max_payload). Returns "" when the job may be sent, else the
// one-line error; never touches /api/execute.
std::string preflightJob(DeviceSession &session, std::size_t crpk_size,
                         const std::vector<std::string> &ops_needed);

// Decode a CRRS body (Section 6.6) into named outputs in flow return order.
// `dev` is the deviceLabel(). Returns "" on success, else a one-line error.
std::string decodeCrrsOutputs(const std::vector<uint8_t> &body,
                              const std::string &dev,
                              const std::vector<std::string> &output_names,
                              RemoteOutputs &outputs);

// Full pipeline: encode CRPK, pre-flight via GET /api/status, submit via
// POST /api/execute?mode=sync, parse + decode the CRRS. Pre-flight failures
// never touch /api/execute.
RemoteRunResult executeRemoteSync(DeviceSession &session, const RemoteJob &job);

// Exact-compare remote-decoded outputs against local interpreter outputs
// (dims and data must match element-for-element). Returns the empty string
// on PASS, else a one-line mismatch description (the --verify mechanism).
std::string verifyAgainstLocal(
    const RemoteOutputs &remote,
    const std::unordered_map<std::string, caramel::interp::Value> &local);

}  // namespace caramel::net

#endif  // CARAMEL_NET_REMOTE_RUN_H
