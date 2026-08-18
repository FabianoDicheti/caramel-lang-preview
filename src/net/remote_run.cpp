// ============================================================================
// Caramel Language - Remote flow execution (CDP client side, sync mode)
// ----------------------------------------------------------------------------
// Ticket:  lang_043 (Remote Execution in caramel-run)
// Version: 1.0.0
// ----------------------------------------------------------------------------
// See include/caramel/net/remote_run.h for the behavioral contract.
// ============================================================================
#include "caramel/net/remote_run.h"

#include <exception>
#include <sstream>

#include "caramel/proto/errors.h"

namespace caramel::net {
namespace {

namespace proto = caramel::proto;
namespace interp = caramel::interp;
namespace ir = caramel::ir;

// One-line rendering of a Value for mismatch messages ("[2x2] [19, 22, ...]"
// or "scalar 5"); mirrors the caramel-run print format, minus the newline.
std::string valueBrief(const interp::Value &v) {
  std::ostringstream os;
  if (v.is_scalar()) {
    os << "scalar " << v.data[0];
    return os.str();
  }
  os << "[";
  for (size_t i = 0; i < v.dims.size(); ++i) {
    if (i) os << "x";
    os << v.dims[i];
  }
  os << "] [";
  for (size_t i = 0; i < v.data.size(); ++i) {
    if (i) os << ", ";
    os << v.data[i];
  }
  os << "]";
  return os.str();
}

std::string joinComma(const std::vector<std::string> &items) {
  std::string out;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i) out += ", ";
    out += items[i];
  }
  return out;
}

}  // namespace

std::string deviceLabel(const DeviceSession &session) {
  return session.config().host + ":" + std::to_string(session.config().port);
}

// Route a DeviceSession failure into the one-message-per-code mapping when a
// Section 7 code was parsed; otherwise keep the session's own message
// (transport/config/protocol failures have no wire code).
std::string sessionErrorText(const DeviceError &err) {
  if (!err.proto_code.empty()) {
    return remoteErrorMessage(err.proto_code, err.proto_detail);
  }
  return err.message;
}

const char *workerMnemonic(ir::OpCode op) {
  switch (op) {
    case ir::OpCode::MATMUL: return "matmul";
    case ir::OpCode::ADD: return "add";
    case ir::OpCode::SUB: return "sub";
    case ir::OpCode::MUL: return "mul";
    case ir::OpCode::DIV: return "div";
    case ir::OpCode::SCALAR_MUL: return "scalar_mul";
    case ir::OpCode::TENSOR_SUM: return "tensor_sum";
    case ir::OpCode::TENSOR_MEAN: return "tensor_mean";
    case ir::OpCode::TENSOR_MAX: return "tensor_max";
    case ir::OpCode::TENSOR_MIN: return "tensor_min";
    case ir::OpCode::GT: return "gt";
    case ir::OpCode::LT: return "lt";
    case ir::OpCode::EQ: return "eq";
    case ir::OpCode::GE: return "ge";
    case ir::OpCode::LE: return "le";
    case ir::OpCode::NE: return "ne";
    case ir::OpCode::AND: return "and";
    case ir::OpCode::OR: return "or";
    case ir::OpCode::XOR: return "xor";
    case ir::OpCode::NOT: return "not";
    case ir::OpCode::RELU: return "relu";
    case ir::OpCode::SIGMOID: return "sigmoid";
    case ir::OpCode::TANH: return "tanh";
    case ir::OpCode::SOFTMAX: return "softmax";
    case ir::OpCode::CONV2D: return "conv2d";
    case ir::OpCode::MAXPOOL2D: return "maxpool2d";
    case ir::OpCode::AVGPOOL2D: return "avgpool2d";
    case ir::OpCode::TRANSPOSE: return "transpose";
    case ir::OpCode::CONCAT: return "concat";
    case ir::OpCode::RESHAPE: return "reshape";
    case ir::OpCode::SPLIT: return "split";
    case ir::OpCode::QUANTIZE: return "quantize";
    case ir::OpCode::DEQUANTIZE: return "dequantize";
    case ir::OpCode::LAYERNORM: return "layernorm";
    case ir::OpCode::BATCHNORM: return "batchnorm";
    // Structural opcodes: part of every IR module, not capability-gated.
    case ir::OpCode::NOP:
    case ir::OpCode::PARAM:
    case ir::OpCode::CONST:
    case ir::OpCode::INPUT:
    case ir::OpCode::SYNC:
    case ir::OpCode::RETURN:
    case ir::OpCode::UNKNOWN:
      return nullptr;
  }
  return nullptr;
}

std::optional<std::vector<std::string>> requiredOps(
    const std::vector<uint8_t> &irBytes) {
  const auto mod = ir::deserialize(irBytes);
  if (!mod) return std::nullopt;
  std::vector<std::string> ops;
  for (const auto &ins : mod->instructions) {
    const char *m = workerMnemonic(ins.opcode);
    if (!m) continue;
    bool seen = false;
    for (const auto &o : ops) {
      if (o == m) {
        seen = true;
        break;
      }
    }
    if (!seen) ops.emplace_back(m);
  }
  return ops;
}

std::string remoteErrorMessage(const std::string &code,
                               const std::string &detail) {
  const std::string suffix =
      detail.empty() ? std::string() : " [worker: " + detail + "]";
  if (code == proto::kErrAuthFailed) {
    return "worker rejected the credentials (AUTH_FAILED): check "
           "--device-user/--device-pass (or CARAMEL_DEVICE_PASS)" + suffix;
  }
  if (code == proto::kErrAuthRequired) {
    return "worker rejected the session token (AUTH_REQUIRED): the worker "
           "rebooted or evicted the session; re-run, or refresh "
           "--device-token" + suffix;
  }
  if (code == proto::kErrProtoVersion) {
    return "worker does not speak CDP protocol v1 (PROTO_VERSION): upgrade "
           "the worker or this caramel-run so both use proto 1" + suffix;
  }
  if (code == proto::kErrBadEnvelope) {
    return "worker could not parse the request envelope (BAD_ENVELOPE): "
           "controller/worker CRPK codec disagree - verify both sides "
           "against the golden fixtures" + suffix;
  }
  if (code == proto::kErrUnsupportedOp) {
    return "worker cannot execute an operation in this flow "
           "(UNSUPPORTED_OP): dispatch to a worker whose `ops` list covers "
           "the flow" + suffix;
  }
  if (code == proto::kErrShapeMismatch) {
    return "input tensor shapes are inconsistent with the flow "
           "(SHAPE_MISMATCH): check the --in dimensions against the flow "
           "parameters" + suffix;
  }
  if (code == proto::kErrTooLarge) {
    return "request exceeds the worker's max_payload (TOO_LARGE): shrink "
           "the inputs or use a worker with a larger limit" + suffix;
  }
  if (code == proto::kErrTooLargeForSync) {
    return "job too large for synchronous execution (TOO_LARGE_FOR_SYNC): "
           "async submission lands with lang_044; shrink the job for now" +
           suffix;
  }
  if (code == proto::kErrBusy) {
    return "worker has no free job slots (BUSY): retry shortly or dispatch "
           "to another worker" + suffix;
  }
  if (code == proto::kErrNotReady) {
    return "result is not ready yet (NOT_READY): unexpected in sync mode - "
           "poll again or report a worker bug" + suffix;
  }
  if (code == proto::kErrNoSuchJob) {
    return "unknown or already-freed job id (NO_SUCH_JOB): results are "
           "read-once and garbage-collected after 300 s" + suffix;
  }
  if (code == proto::kErrExecFailed) {
    return "worker runtime failure during execution (EXEC_FAILED): check "
           "value ranges (i8 outputs must fit [-128, 127]; accumulations "
           "must fit i32)" + suffix;
  }
  return "worker returned unrecognized error code \"" + code + "\"" + suffix;
}

std::string encodeJobCrpk(const RemoteJob &job, std::vector<uint8_t> &crpk,
                          std::vector<std::string> &ops_needed) {
  proto::CrpkRequest req;
  req.quant = job.quant;
  req.ir = job.ir;
  for (size_t i = 0; i < job.inputs.size(); ++i) {
    if (i > 255) {
      return "flow has more than 256 inputs (CRPK param_index is u8)";
    }
    auto entry = proto::tensorFromValue(static_cast<uint8_t>(i),
                                        job.quant.dtype, job.inputs[i].second);
    if (!entry) {
      return "input '" + job.inputs[i].first +
             "' cannot be encoded for remote execution (rank must be "
             "1..4 and every element must fit the quantized dtype)";
    }
    req.inputs.push_back(std::move(*entry));
  }

  try {
    crpk = proto::serializeCrpk(req);
  } catch (const std::exception &e) {
    return std::string("cannot serialize CRPK request: ") + e.what();
  }

  const auto ops = requiredOps(job.ir);
  if (!ops) {
    return "compiled flow is not a valid binary IR module";
  }
  ops_needed = *ops;
  return std::string();
}

std::string preflightJob(DeviceSession &session, std::size_t crpk_size,
                         const std::vector<std::string> &ops_needed) {
  const std::string dev = deviceLabel(session);
  auto st = session.status();
  if (!st.ok()) {
    return sessionErrorText(st.error);
  }
  if (st.status->proto != 1) {
    return "pre-flight failed (PROTO_VERSION): worker " + dev +
           " speaks proto " + std::to_string(st.status->proto) +
           ", this controller requires proto 1; not sending the job";
  }
  std::vector<std::string> missing;
  for (const auto &op : ops_needed) {
    bool found = false;
    for (const auto &have : st.status->ops) {
      if (have == op) {
        found = true;
        break;
      }
    }
    if (!found) missing.push_back(op);
  }
  if (!missing.empty()) {
    return "pre-flight failed (UNSUPPORTED_OP): worker " + dev +
           " does not support: " + joinComma(missing) +
           "; not sending the job";
  }
  if (crpk_size > st.status->max_payload) {
    return "pre-flight failed (TOO_LARGE): CRPK request is " +
           std::to_string(crpk_size) + " bytes but worker " + dev +
           " accepts at most " + std::to_string(st.status->max_payload) +
           "; not sending the job";
  }
  return std::string();
}

std::string decodeCrrsOutputs(const std::vector<uint8_t> &body,
                              const std::string &dev,
                              const std::vector<std::string> &output_names,
                              RemoteOutputs &outputs) {
  const auto crrs = proto::parseCrrs(body);
  if (!crrs) {
    return "malformed CRRS response from worker " + dev;
  }
  if (crrs->status != 0) {
    return "worker " + dev + " reported CRRS status " +
           std::to_string(crrs->status) + " (nonzero = execution error)";
  }
  if (crrs->outputs.size() != output_names.size()) {
    return "worker " + dev + " returned " +
           std::to_string(crrs->outputs.size()) +
           " output tensor(s), expected " +
           std::to_string(output_names.size());
  }

  RemoteOutputs decoded;
  for (size_t i = 0; i < crrs->outputs.size(); ++i) {
    const auto &entry = crrs->outputs[i];
    if (entry.param_index != i) {
      return "CRRS output table from worker " + dev +
             " is out of order (entry " + std::to_string(i) +
             " has index " + std::to_string(entry.param_index) + ")";
    }
    auto v = proto::valueFromTensor(entry);
    if (!v) {
      return "CRRS output '" + output_names[i] + "' from worker " + dev +
             " cannot be decoded";
    }
    decoded.emplace_back(output_names[i], std::move(*v));
  }
  outputs = std::move(decoded);
  return std::string();
}

RemoteRunResult executeRemoteSync(DeviceSession &session,
                                  const RemoteJob &job) {
  RemoteRunResult out;
  const std::string dev = deviceLabel(session);

  // --- Encode the CRPK envelope ------------------------------------------
  std::vector<uint8_t> crpk;
  std::vector<std::string> ops_needed;
  out.error = encodeJobCrpk(job, crpk, ops_needed);
  if (!out.error.empty()) return out;

  // --- Pre-flight: GET /api/status (never touches /api/execute) ----------
  out.error = preflightJob(session, crpk.size(), ops_needed);
  if (!out.error.empty()) return out;

  // --- Submit: POST /api/execute?mode=sync (Section 6.2) ------------------
  auto res = session.authedRequest("POST", "/api/execute?mode=sync", crpk,
                                   "application/x-caramel-crpk");
  if (!res.ok()) {
    out.error = sessionErrorText(res.error);
    return out;
  }
  const HttpResponse &resp = *res.response;

  if (resp.status == 202) {
    out.error = "worker " + dev + " queued the job asynchronously despite "
                "mode=sync; async polling is out of scope until lang_044";
    return out;
  }
  if (resp.status != 200) {
    const auto pe = proto::parseErrorBody(resp.body);
    if (pe) {
      out.error = remoteErrorMessage(pe->code, pe->detail);
    } else {
      out.error = "worker " + dev + " returned HTTP " +
                  std::to_string(resp.status) +
                  " with no parsable CDP error body";
    }
    return out;
  }

  // --- Decode the CRRS result (Section 6.6) -------------------------------
  RemoteOutputs outputs;
  out.error = decodeCrrsOutputs(resp.body, dev, job.output_names, outputs);
  if (!out.error.empty()) return out;
  out.outputs = std::move(outputs);
  return out;
}

std::string verifyAgainstLocal(
    const RemoteOutputs &remote,
    const std::unordered_map<std::string, interp::Value> &local) {
  for (const auto &kv : remote) {
    const auto it = local.find(kv.first);
    if (it == local.end()) {
      return "local run produced no output named '" + kv.first + "'";
    }
    // Shapes must match, with one exception: CRPK tensors always carry rank
    // >= 1, so a value the interpreter holds as a rank-0 scalar comes back off
    // the wire as rank-1 [1]. Whole-tensor reductions hit this on every run
    // (kernel/ir_exec.c commits tensor_sum/mean/max/min to dims {1}), which
    // reported a MISMATCH on identical data. Treat only that exact pair as
    // equivalent — any other shape difference is still a real difference.
    const auto &lv = it->second, &rv = kv.second;
    const bool scalar_shape_equiv =
        (lv.dims.empty() && rv.dims.size() == 1 && rv.dims[0] == 1) ||
        (rv.dims.empty() && lv.dims.size() == 1 && lv.dims[0] == 1);
    if ((lv.dims != rv.dims && !scalar_shape_equiv) || lv.data != rv.data) {
      return "output '" + kv.first + "' differs: local " +
             valueBrief(it->second) + " vs remote " + valueBrief(kv.second);
    }
  }
  return std::string();
}

}  // namespace caramel::net
