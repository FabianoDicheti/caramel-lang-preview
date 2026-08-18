// ============================================================================
// Caramel Language - Async jobs client implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_044 (Async Jobs Client + device:: Syntax in .crml)
// Version: 1.0.0
// ----------------------------------------------------------------------------
// See include/caramel/net/jobs.h for the behavioral contract.
// ============================================================================
#include "caramel/net/jobs.h"

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "caramel/proto/errors.h"
#include "caramel/proto/json_lite.h"

namespace caramel::net {
namespace {

namespace proto = caramel::proto;

void doSleep(const JobsOptions &opts, int ms) {
  if (opts.sleep) {
    opts.sleep(ms);
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }
}

std::string bodyText(const std::vector<uint8_t> &body) {
  return std::string(body.begin(), body.end());
}

// The distinct "read-once result is gone" message (Section 6.4: results are
// read-once and garbage-collected after 300 s; a worker reboot also drops
// live jobs). Must be unmistakable: the job may have completed, but the
// output is unrecoverable - only a re-submit helps.
std::string resultLostMessage(int64_t job_id, const std::string &dev) {
  return "job " + std::to_string(job_id) + " result lost on worker " + dev +
         " (NO_SUCH_JOB): results are read-once and expire after 300 s - "
         "re-submit the job";
}

// Non-2xx handling shared by submit/poll/fetch: prefer the Section 7 error
// body mapping, fall back to a generic HTTP line.
std::string httpErrorMessage(const HttpResponse &resp, const std::string &dev) {
  const auto pe = proto::parseErrorBody(resp.body);
  if (pe) return remoteErrorMessage(pe->code, pe->detail);
  return "worker " + dev + " returned HTTP " + std::to_string(resp.status) +
         " with no parsable CDP error body";
}

// The async pipeline after CRPK encoding (shared by executeRemoteAsync and
// the large-payload branch of executeRemoteAuto). Pre-flight is included.
RemoteRunResult runAsyncWithCrpk(DeviceSession &session, const RemoteJob &job,
                                 const std::vector<uint8_t> &crpk,
                                 const std::vector<std::string> &ops_needed,
                                 const JobsOptions &opts) {
  RemoteRunResult out;
  const std::string dev = deviceLabel(session);

  // --- Pre-flight: GET /api/status (never touches /api/execute) ----------
  out.error = preflightJob(session, crpk.size(), ops_needed);
  if (!out.error.empty()) return out;

  // --- Submit: POST /api/execute, async (Section 6.2) ---------------------
  auto res = session.authedRequest("POST", "/api/execute", crpk,
                                   "application/x-caramel-crpk");
  if (!res.ok()) {
    out.error = sessionErrorText(res.error);
    return out;
  }
  {
    const HttpResponse &resp = *res.response;
    if (resp.status == 200) {
      // Spec: sync-only workers answer async submissions with 501, never
      // with an inline 200 result.
      out.error = "worker " + dev +
                  " answered an async submission with HTTP 200 (expected "
                  "202 + job id); treating this as a protocol error";
      return out;
    }
    if (resp.status != 202) {
      out.error = httpErrorMessage(resp, dev);
      return out;
    }
  }
  int64_t job_id = -1;
  {
    const auto obj = proto::parseJsonObject(bodyText(res.response->body));
    const auto id = obj ? obj->getInt("job") : std::nullopt;
    if (!id || *id < 0) {
      out.error = "worker " + dev +
                  " accepted the job (HTTP 202) but sent no parsable job id";
      return out;
    }
    job_id = *id;
  }
  const std::string jid = std::to_string(job_id);

  // --- Poll: GET /api/job?id=N with exponential backoff (Section 6.3) -----
  // First poll immediately; sleep (through the injectable hook) only between
  // polls. `waited` accumulates INTENDED sleep, giving a deterministic
  // timeout under a mocked clock.
  int delay = opts.poll_initial_ms;
  long long waited = 0;
  for (;;) {
    auto poll = session.authedRequest("GET", "/api/job?id=" + jid);
    if (!poll.ok()) {
      out.error = sessionErrorText(poll.error);
      return out;
    }
    const HttpResponse &resp = *poll.response;
    if (resp.status == 404) {
      out.error = resultLostMessage(job_id, dev);
      return out;
    }
    if (resp.status != 200) {
      out.error = httpErrorMessage(resp, dev);
      return out;
    }
    const auto obj = proto::parseJsonObject(bodyText(resp.body));
    const auto state = obj ? obj->getString("state") : std::nullopt;
    if (!state) {
      out.error = "worker " + dev + " sent a job status for job " + jid +
                  " without a parsable state";
      return out;
    }
    if (*state == "done") break;
    if (*state == "queued" || *state == "running") {
      if (waited >= opts.max_wait_ms) {
        out.error = "timed out after " + std::to_string(waited) +
                    " ms waiting for job " + jid + " on worker " + dev +
                    "; the worker garbage-collects unfetched results after "
                    "300 s - re-submit if the output is still needed";
        return out;
      }
      doSleep(opts, delay);
      waited += delay;
      delay = delay * 2;
      if (delay > opts.poll_max_ms) delay = opts.poll_max_ms;
      continue;
    }
    if (*state == "error") {
      const auto code = obj->getString("code");
      const auto detail = obj->getString("detail");
      if (code && !code->empty()) {
        out.error = remoteErrorMessage(*code, detail ? *detail : "");
      } else {
        out.error = "job " + jid + " on worker " + dev +
                    " failed without an error code";
      }
      return out;
    }
    if (*state == "cancelled") {
      out.error = "job " + jid + " on worker " + dev +
                  " was cancelled before it finished";
      return out;
    }
    out.error = "worker " + dev + " reported unknown state \"" + *state +
                "\" for job " + jid;
    return out;
  }

  // --- Fetch: GET /api/result?id=N (read-once, Section 6.4) ---------------
  auto fetch = session.authedRequest("GET", "/api/result?id=" + jid);
  if (!fetch.ok()) {
    out.error = sessionErrorText(fetch.error);
    return out;
  }
  const HttpResponse &resp = *fetch.response;
  if (resp.status == 404) {
    out.error = resultLostMessage(job_id, dev);
    return out;
  }
  if (resp.status != 200) {
    out.error = httpErrorMessage(resp, dev);
    return out;
  }

  // Decode BEFORE reporting success: the fetch consumed the read-once
  // result, so a decode failure here is final (and the message says which
  // stage failed).
  RemoteOutputs outputs;
  out.error = decodeCrrsOutputs(resp.body, dev, job.output_names, outputs);
  if (!out.error.empty()) return out;
  out.outputs = std::move(outputs);
  return out;
}

}  // namespace

RemoteRunResult executeRemoteAsync(DeviceSession &session,
                                   const RemoteJob &job,
                                   const JobsOptions &opts) {
  RemoteRunResult out;
  std::vector<uint8_t> crpk;
  std::vector<std::string> ops_needed;
  out.error = encodeJobCrpk(job, crpk, ops_needed);
  if (!out.error.empty()) return out;
  return runAsyncWithCrpk(session, job, crpk, ops_needed, opts);
}

RemoteRunResult executeRemoteAuto(DeviceSession &session, const RemoteJob &job,
                                  const JobsOptions &opts) {
  RemoteRunResult out;
  std::vector<uint8_t> crpk;
  std::vector<std::string> ops_needed;
  out.error = encodeJobCrpk(job, crpk, ops_needed);
  if (!out.error.empty()) return out;
  if (crpk.size() < opts.sync_threshold) {
    // Small job: one round trip, no polling. executeRemoteSync re-encodes
    // the (small, cheap) envelope so the lang_043 path stays byte-for-byte
    // untouched.
    return executeRemoteSync(session, job);
  }
  return runAsyncWithCrpk(session, job, crpk, ops_needed, opts);
}

}  // namespace caramel::net
