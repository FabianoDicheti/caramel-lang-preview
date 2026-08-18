// ============================================================================
// Caramel Language - Async jobs client (CDP client side)
// ----------------------------------------------------------------------------
// Ticket:   lang_044 (Async Jobs Client + device:: Syntax in .crml)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// The asynchronous half of remote execution (PROTOCOL_SPEC.md Sections
// 6.2-6.4), built from the same pipeline stages as executeRemoteSync
// (net/remote_run.h):
//
//   * Submit via POST /api/execute (NO mode parameter) and expect
//     202 {"job": N, "state": "queued"}. Any 200 to an async submission is
//     a protocol error (per the spec, a sync-only worker answers 501
//     NOT_IMPLEMENTED instead).
//   * Poll GET /api/job?id=N with exponential backoff: first poll is
//     immediate, then sleeps of 50 ms doubling to a 1 s cap (constants
//     below). All sleeping goes through JobsOptions::sleep so tests can
//     inject a recording no-op clock; the accumulated *intended* sleep time
//     is compared against max_wait_ms, which makes the timeout deterministic
//     under a mocked clock.
//   * On "done", fetch GET /api/result?id=N. Results are read-once with a
//     300 s TTL on the worker, so the CRRS body is fully decoded BEFORE the
//     call reports success - a decode failure after the fetch cannot be
//     retried and says so.
//   * A 404/NO_SUCH_JOB on poll or result fetch (job expired, worker
//     rebooted, or the result TTL passed) yields a distinct "result lost
//     ... re-submit" error instead of a hang or a generic message.
//   * Job states: queued|running -> keep polling; done -> fetch result;
//     error -> map the embedded "code"/"detail" through remoteErrorMessage;
//     cancelled / unknown states -> distinct errors.
//
// executeRemoteAuto() picks the transport: payloads whose CRPK envelope is
// strictly smaller than kSyncThresholdBytes (64 KiB) use ?mode=sync
// transparently (one round trip, no polling); everything else goes async.
//
// Error handling follows the repo idiom (net/remote_run.h): no exceptions
// cross the API; RemoteRunResult carries either decoded outputs or a
// one-line error ready for stderr. Secrets never appear in any message.
// ============================================================================
#ifndef CARAMEL_NET_JOBS_H
#define CARAMEL_NET_JOBS_H

#include <cstddef>
#include <functional>

#include "caramel/net/remote_run.h"

namespace caramel::net {

// executeRemoteAuto(): CRPK envelopes strictly smaller than this go through
// the sync path (?mode=sync); larger ones are submitted asynchronously.
// 64 KiB: well under any spec-compliant max_payload example (1 MiB) while
// keeping small interactive jobs on the cheap single-round-trip path.
inline constexpr std::size_t kSyncThresholdBytes = 64 * 1024;

// Poll backoff (Section 6.3 polling loop): 50 ms doubling to a 1 s cap.
inline constexpr int kPollInitialMs = 50;
inline constexpr int kPollMaxMs = 1000;

// Give up polling after this much accumulated intended sleep. Matches the
// worker's own 300 s result-retention TTL (Section 6.4).
inline constexpr int kDefaultMaxWaitMs = 300 * 1000;

// Tunables + injectable clock for the async path. Defaults reproduce
// production behavior; tests override `sleep` to record delays instead of
// actually sleeping (the poll loop never sleeps except through this hook).
struct JobsOptions {
  std::size_t sync_threshold = kSyncThresholdBytes;  // executeRemoteAuto only
  int poll_initial_ms = kPollInitialMs;
  int poll_max_ms = kPollMaxMs;
  int max_wait_ms = kDefaultMaxWaitMs;
  std::function<void(int ms)> sleep;  // null -> std::this_thread::sleep_for
};

// Async pipeline: encode CRPK, pre-flight via GET /api/status, submit via
// POST /api/execute (202 + job id), poll /api/job, fetch + decode
// /api/result. Pre-flight failures never touch /api/execute.
RemoteRunResult executeRemoteAsync(DeviceSession &session,
                                   const RemoteJob &job,
                                   const JobsOptions &opts = JobsOptions{});

// Transport auto-selection: sync for CRPK envelopes strictly smaller than
// opts.sync_threshold, async otherwise. This is the entry point the
// dispatcher (net/dispatch.h) uses.
RemoteRunResult executeRemoteAuto(DeviceSession &session,
                                  const RemoteJob &job,
                                  const JobsOptions &opts = JobsOptions{});

}  // namespace caramel::net

#endif  // CARAMEL_NET_JOBS_H
