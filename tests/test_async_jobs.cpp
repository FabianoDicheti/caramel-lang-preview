// ============================================================================
// Caramel Language - Async jobs client tests (lang_044)
// ----------------------------------------------------------------------------
// Ticket: lang_044 (Async Jobs Client + device:: Syntax in .crml)
// ----------------------------------------------------------------------------
// The async half of remote execution (net/jobs.h) against a scripted stub
// worker (tests/stub_server.h), PROTOCOL_SPEC.md Sections 6.2-6.4 + 7:
//   * happy path: POST /api/execute (no mode param) -> 202 + job id, poll
//     GET /api/job through queued/running to done, fetch + decode
//     GET /api/result; outputs byte-match the local interpreter run
//   * backoff: 50 ms doubling to the 1 s cap, observed through an injected
//     recording sleep hook (mock clock - the test never really sleeps)
//   * timeout: bounded by max_wait_ms (deterministic under the mock clock),
//     never a hang
//   * TTL/read-once: 404/NO_SUCH_JOB on poll or on result fetch surfaces the
//     distinct "result lost ... re-submit" error
//   * a 200 answer to an async submission is a protocol error (spec 6.2)
//   * job state "error" maps its code/detail through remoteErrorMessage;
//     "cancelled" and unknown states get distinct errors
//   * executeRemoteAuto: CRPK under the 64 KiB threshold goes ?mode=sync in
//     one round trip (no /api/job traffic); above the threshold it submits
//     async
//   * a result that fetches but fails to decode reports the decode failure
//     (the read-once fetch cannot be retried)
// ============================================================================
#include "caramel/net/jobs.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "caramel/interp/interpreter.h"
#include "caramel/interp/lambda.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/binary_ir.h"
#include "caramel/ir/dataflow.h"
#include "caramel/net/device.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"
#include "caramel/proto/crpk.h"
#include "caramel/proto/errors.h"
#include "stub_server.h"

using namespace caramel::net;
using namespace caramel::teststub;
namespace interp = caramel::interp;
namespace ir = caramel::ir;
namespace proto = caramel::proto;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// ---------------------------------------------------------------------------
// The canonical matmul_relu flow (same compile path as test_remote_execute)
// ---------------------------------------------------------------------------

static const char kMatmulReluSrc[] =
    "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
    "calc::lambda_flow layer(x, w, bias) {\n"
    "    x w matmul p =\n"
    "    p bias elemwise_add s =\n"
    "    s relu result =\n"
    "} return result;\n";

struct Compiled {
  std::unique_ptr<caramel::ast::Program> program;  // keeps AST alive
  ir::DataflowGraph graph;
  std::vector<std::string> returns;
};

static Compiled compileLayer() {
  Compiled c;
  caramel::parse::Lexer lx(kMatmulReluSrc);
  caramel::parse::Parser p(lx.tokenize());
  c.program = p.parse();
  CHECK(p.ok());
  for (auto &it : c.program->items) {
    if (it->kind == caramel::ast::NodeKind::LambdaFlow) {
      auto *lf = static_cast<caramel::ast::LambdaFlow *>(it.get());
      c.graph = ir::buildDataflow(*lf);
      c.returns = lf->returns;
      break;
    }
  }
  return c;
}

static std::vector<std::pair<std::string, interp::Value>> canonicalInputs() {
  return {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
      {"w", interp::Value::tensor({2, 2}, {5, 6, 7, 8})},
      {"bias", interp::Value::tensor({2, 2}, {0, 0, -100, 0})},
  };
}

static RemoteJob buildJob(const Compiled &c) {
  RemoteJob job;
  job.quant.quantres = 0;
  job.quant.dtype = proto::kDTypeInt32;
  job.quant.quantmin = -1000;
  job.quant.quantmax = 1000;
  job.ir = ir::serialize(c.graph);
  job.inputs = canonicalInputs();
  job.output_names = c.returns;
  return job;
}

// The correct result CRRS for the canonical inputs: relu(x*w + bias) =
// [19, 22, 0, 50], built with the lang_040 codec (no fixture dependency).
static std::vector<uint8_t> goodCrrs() {
  proto::CrrsResponse resp;
  const auto t = proto::tensorFromValue(
      0, proto::kDTypeInt32, interp::Value::tensor({2, 2}, {19, 22, 0, 50}));
  CHECK(t.has_value());
  resp.outputs.push_back(*t);
  return proto::buildCrrs(resp);
}

// ---------------------------------------------------------------------------
// Async-capable stub worker (spec Sections 5, 6.1, 6.2, 6.3, 6.4, 7)
// ---------------------------------------------------------------------------

static const char kToken[] = "0123456789abcdef0123456789abcdef";
static const char kJsonHdr[] = "Content-Type: application/json\r\n";
static const char kCrrsHdr[] =
    "Content-Type: application/x-caramel-crrs\r\n";
static const char kDefaultOpsJson[] =
    "[\"matmul\",\"add\",\"sub\",\"mul\",\"relu\",\"transpose\"]";

static std::string jsonError(const std::string &code,
                             const std::string &detail) {
  return "{\"error\":\"" + code + "\",\"detail\":\"" + detail + "\"}";
}

struct AsyncWorkerOptions {
  // Async submit answer (default: accepted as job 7).
  int exec_status = 202;
  std::string exec_content_type = "application/json";
  std::string exec_body = "{\"job\":7,\"state\":\"queued\"}";
  // GET /api/job?id=7 answers, consumed in order; the LAST entry repeats
  // once the sequence is exhausted.
  std::vector<std::pair<int, std::string>> polls;
  // GET /api/result?id=7 answer.
  int result_status = 200;
  std::string result_content_type = "application/x-caramel-crrs";
  std::string result_body;
};

static std::string statusJson() {
  return std::string("{\"proto\":1,\"alias\":\"stub\",\"kind\":\"x86\","
                     "\"uptime_s\":1,\"requests\":1,\"jobs_running\":0,"
                     "\"jobs_capacity\":4,\"ops\":") +
         kDefaultOpsJson + ",\"max_payload\":1048576}";
}

// Handler state shared across connections (each request = one connection).
static StubServer::Handler asyncWorkerHandler(
    std::shared_ptr<AsyncWorkerOptions> opt,
    std::shared_ptr<std::atomic<size_t>> poll_count) {
  return [opt, poll_count](int fd, const std::string &raw) {
    if (raw.rfind("GET /api/status HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, 200, kJsonHdr, statusJson());
    } else if (raw.rfind("POST /api/auth HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, 200, kJsonHdr,
                   std::string("{\"token\":\"") + kToken +
                       "\",\"alias\":\"stub\"}");
    } else if (raw.rfind("POST /api/execute HTTP/1.1\r\n", 0) == 0) {
      // The ASYNC submit: exact path, no ?mode=sync query.
      sendResponse(fd, opt->exec_status,
                   "Content-Type: " + opt->exec_content_type + "\r\n",
                   opt->exec_body);
    } else if (raw.rfind("GET /api/job?id=7 HTTP/1.1\r\n", 0) == 0) {
      const size_t i = poll_count->fetch_add(1);
      const size_t idx = i < opt->polls.size() ? i : opt->polls.size() - 1;
      sendResponse(fd, opt->polls[idx].first, kJsonHdr,
                   opt->polls[idx].second);
    } else if (raw.rfind("GET /api/result?id=7 HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, opt->result_status,
                   "Content-Type: " + opt->result_content_type + "\r\n",
                   opt->result_body);
    } else {
      sendResponse(fd, 404, kJsonHdr,
                   jsonError(proto::kErrNoSuchJob, "unexpected route"));
    }
  };
}

static DeviceSession makeSession(uint16_t port) {
  DeviceConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = port;
  cfg.user = "fabiano";
  cfg.pass = "sw0rdfish-9";
  return DeviceSession(cfg);
}

// Recording mock clock: JobsOptions whose sleep hook appends the requested
// delay instead of sleeping (the poll loop never sleeps except through the
// hook, so these tests are instantaneous and deterministic).
static JobsOptions recordingOpts(std::shared_ptr<std::vector<int>> sleeps) {
  JobsOptions o;
  o.sleep = [sleeps](int ms) { sleeps->push_back(ms); };
  return o;
}

static size_t countPrefix(const std::vector<std::string> &reqs,
                          const std::string &prefix) {
  size_t n = 0;
  for (const auto &r : reqs)
    if (r.rfind(prefix, 0) == 0) ++n;
  return n;
}

// ---------------------------------------------------------------------------
// (a) Happy path: submit -> poll (queued/running) -> done -> fetch + decode
// ---------------------------------------------------------------------------

static void test_async_happy_path() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {
      {200, "{\"job\":7,\"state\":\"queued\"}"},
      {200, "{\"job\":7,\"state\":\"running\"}"},
      {200, "{\"job\":7,\"state\":\"running\"}"},
      {200, "{\"job\":7,\"state\":\"done\"}"},
  };
  const auto crrs = goodCrrs();
  opt->result_body.assign(crrs.begin(), crrs.end());
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/10));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(res.ok());
  CHECK(res.outputs->size() == 1);
  CHECK((*res.outputs)[0].first == "result");
  const interp::Value &v = (*res.outputs)[0].second;
  CHECK(v.dims == (std::vector<int64_t>{2, 2}));
  CHECK(v.data == (std::vector<int64_t>{19, 22, 0, 50}));

  // Byte-exact agreement with the local interpreter.
  interp::Interpreter vm;
  vm.add_evaluator(interp::matrixOpEvaluator());
  vm.add_evaluator(interp::lambdaOpEvaluator());
  for (const auto &kv : canonicalInputs()) vm.set_input(kv.first, kv.second);
  auto local = vm.run(c.graph);
  CHECK(local.ok());
  CHECK(verifyAgainstLocal(*res.outputs, local.outputs).empty());

  // Wire sequence: status, auth, ASYNC execute (no mode param), 4 polls,
  // one result fetch - 8 requests total.
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 8);
  CHECK(reqs[0].rfind("GET /api/status HTTP/1.1\r\n", 0) == 0);
  CHECK(reqs[1].rfind("POST /api/auth HTTP/1.1\r\n", 0) == 0);
  CHECK(reqs[2].rfind("POST /api/execute HTTP/1.1\r\n", 0) == 0);
  CHECK(reqs[2].find("mode=sync") == std::string::npos);
  CHECK(reqs[2].find("Content-Type: application/x-caramel-crpk\r\n") !=
        std::string::npos);
  CHECK(countPrefix(reqs, "GET /api/job?id=7 HTTP/1.1\r\n") == 4);
  CHECK(reqs[7].rfind("GET /api/result?id=7 HTTP/1.1\r\n", 0) == 0);
  CHECK(reqs[7].find(std::string("X-Caramel-Token: ") + kToken + "\r\n") !=
        std::string::npos);

  // Backoff between the 4 polls: first poll immediate, then 50, 100, 200.
  CHECK(*sleeps == (std::vector<int>{50, 100, 200}));
}

// ---------------------------------------------------------------------------
// (b) Backoff: 50 ms doubling to the 1 s cap (mock clock, no real sleeping)
// ---------------------------------------------------------------------------

static void test_backoff_doubles_and_caps() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  for (int i = 0; i < 8; ++i)
    opt->polls.push_back({200, "{\"job\":7,\"state\":\"running\"}"});
  opt->polls.push_back({200, "{\"job\":7,\"state\":\"done\"}"});
  const auto crrs = goodCrrs();
  opt->result_body.assign(crrs.begin(), crrs.end());
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/16));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(res.ok());
  // 8 not-done polls -> 8 sleeps: 50 doubling, clamped at the 1000 ms cap.
  CHECK(*sleeps ==
        (std::vector<int>{50, 100, 200, 400, 800, 1000, 1000, 1000}));
}

// ---------------------------------------------------------------------------
// (c) Timeout: bounded by max_wait_ms, deterministic, never a hang
// ---------------------------------------------------------------------------

static void test_poll_timeout_is_bounded() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {{200, "{\"job\":7,\"state\":\"running\"}"}};  // repeats forever
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/12));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  JobsOptions o = recordingOpts(sleeps);
  o.max_wait_ms = 500;  // intended-sleep budget: 50+100+200+400 = 750 >= 500
  auto res = executeRemoteAsync(s, buildJob(c), o);
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("timed out") != std::string::npos);
  CHECK(res.error.find("750 ms") != std::string::npos);  // accumulated waits
  CHECK(res.error.find("re-submit") != std::string::npos);
  CHECK(*sleeps == (std::vector<int>{50, 100, 200, 400}));
  // 5 polls: 4 that slept afterwards + the one that hit the budget.
  CHECK(countPrefix(srv.requests(), "GET /api/job?id=7 HTTP/1.1\r\n") == 5);
}

// ---------------------------------------------------------------------------
// (d) TTL / read-once: 404 surfaces the distinct "result lost" error
// ---------------------------------------------------------------------------

static void test_expired_job_on_poll_is_result_lost() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {{404, jsonError(proto::kErrNoSuchJob, "job expired")}};
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/6));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("result lost") != std::string::npos);
  CHECK(res.error.find("re-submit") != std::string::npos);
  CHECK(res.error.find(proto::kErrNoSuchJob) != std::string::npos);
  CHECK(res.error.find("job 7") != std::string::npos);
  // Distinct from the generic NO_SUCH_JOB mapping: it explains read-once.
  CHECK(res.error != remoteErrorMessage(proto::kErrNoSuchJob, "job expired"));
  CHECK(res.error.find("read-once") != std::string::npos);
  CHECK(sleeps->empty());  // failed fast: no backoff, no hang
}

static void test_expired_result_fetch_is_result_lost() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {{200, "{\"job\":7,\"state\":\"done\"}"}};
  opt->result_status = 404;
  opt->result_content_type = "application/json";
  opt->result_body = jsonError(proto::kErrNoSuchJob, "result expired (TTL)");
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/8));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("result lost") != std::string::npos);
  CHECK(res.error.find("re-submit") != std::string::npos);
  CHECK(res.error.find("300 s") != std::string::npos);  // names the TTL
}

// ---------------------------------------------------------------------------
// (e) Protocol errors around submission and job state
// ---------------------------------------------------------------------------

static void test_sync_200_answer_to_async_submit_is_protocol_error() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->exec_status = 200;  // spec 6.2: a sync-only worker must answer 501
  opt->exec_content_type = "application/x-caramel-crrs";
  const auto crrs = goodCrrs();
  opt->exec_body.assign(crrs.begin(), crrs.end());
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/6));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("202") != std::string::npos);
  CHECK(res.error.find("protocol error") != std::string::npos);
}

static void test_submit_busy_maps_section7_code() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->exec_status = 503;
  opt->exec_body = jsonError(proto::kErrBusy, "no free slots");
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/6));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error == remoteErrorMessage(proto::kErrBusy, "no free slots"));
}

static void test_job_error_state_maps_embedded_code() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {{200,
                 "{\"job\":7,\"state\":\"error\",\"code\":\"EXEC_FAILED\","
                 "\"detail\":\"i8 overflow at instr 4\"}"}};
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/6));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error ==
        remoteErrorMessage(proto::kErrExecFailed, "i8 overflow at instr 4"));
}

static void test_job_cancelled_state_is_distinct() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {{200, "{\"job\":7,\"state\":\"cancelled\"}"}};
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/6));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("cancelled") != std::string::npos);
  CHECK(res.error.find("job 7") != std::string::npos);
}

static void test_undecodable_result_reports_decode_failure() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {{200, "{\"job\":7,\"state\":\"done\"}"}};
  opt->result_body = "XXXX not a crrs";  // fetched (read-once) but undecodable
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/8));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("malformed CRRS") != std::string::npos);
}

// ---------------------------------------------------------------------------
// (f) executeRemoteAuto: sync under the 64 KiB threshold, async above it
// ---------------------------------------------------------------------------

static void test_auto_small_payload_uses_sync() {
  CHECK(kSyncThresholdBytes == 64 * 1024);  // documented threshold constant
  // A sync-only stub: it would fail any async submit or poll route.
  auto opt = std::make_shared<AsyncWorkerOptions>();
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  const auto crrs = goodCrrs();
  const std::string crrs_body(crrs.begin(), crrs.end());
  StubServer srv;
  CHECK(srv.start(
      [crrs_body](int fd, const std::string &raw) {
        if (raw.rfind("GET /api/status HTTP/1.1\r\n", 0) == 0) {
          sendResponse(fd, 200, kJsonHdr, statusJson());
        } else if (raw.rfind("POST /api/auth HTTP/1.1\r\n", 0) == 0) {
          sendResponse(fd, 200, kJsonHdr,
                       std::string("{\"token\":\"") + kToken +
                           "\",\"alias\":\"stub\"}");
        } else if (raw.rfind("POST /api/execute?mode=sync HTTP/1.1\r\n", 0) ==
                   0) {
          sendResponse(fd, 200, kCrrsHdr, crrs_body);
        } else {
          sendResponse(fd, 501,
                       kJsonHdr, jsonError("NOT_IMPLEMENTED",
                                           "async in x86_044"));
        }
      },
      /*connections=*/6));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  // The matmul_relu CRPK is ~205 bytes, far under the 64 KiB default.
  auto res = executeRemoteAuto(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(res.ok());
  CHECK((*res.outputs)[0].second.data ==
        (std::vector<int64_t>{19, 22, 0, 50}));
  const auto reqs = srv.requests();
  CHECK(countPrefix(reqs, "POST /api/execute?mode=sync HTTP/1.1\r\n") == 1);
  CHECK(countPrefix(reqs, "POST /api/execute HTTP/1.1\r\n") == 0);
  CHECK(countPrefix(reqs, "GET /api/job") == 0);  // no polling round trips
  CHECK(countPrefix(reqs, "GET /api/result") == 0);
  CHECK(sleeps->empty());
}

static void test_auto_large_payload_goes_async() {
  auto opt = std::make_shared<AsyncWorkerOptions>();
  opt->polls = {{200, "{\"job\":7,\"state\":\"done\"}"}};
  const auto crrs = goodCrrs();
  opt->result_body.assign(crrs.begin(), crrs.end());
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(asyncWorkerHandler(opt, polls), /*connections=*/8));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  JobsOptions o = recordingOpts(sleeps);
  o.sync_threshold = 1;  // force every payload over the "small" line
  auto res = executeRemoteAuto(s, buildJob(c), o);
  srv.stop();

  CHECK(res.ok());
  const auto reqs = srv.requests();
  CHECK(countPrefix(reqs, "POST /api/execute HTTP/1.1\r\n") == 1);
  CHECK(countPrefix(reqs, "POST /api/execute?mode=sync") == 0);
  CHECK(countPrefix(reqs, "GET /api/job?id=7 HTTP/1.1\r\n") == 1);
  CHECK(countPrefix(reqs, "GET /api/result?id=7 HTTP/1.1\r\n") == 1);
}

// ---------------------------------------------------------------------------
// (g) Pre-flight still guards the async path (never touches /api/execute)
// ---------------------------------------------------------------------------

static void test_async_preflight_unsupported_op() {
  auto polls = std::make_shared<std::atomic<size_t>>(0);
  StubServer srv;
  CHECK(srv.start(
      [](int fd, const std::string &raw) {
        if (raw.rfind("GET /api/status HTTP/1.1\r\n", 0) == 0) {
          sendResponse(fd, 200, kJsonHdr,
                       "{\"proto\":1,\"alias\":\"stub\",\"kind\":\"x86\","
                       "\"uptime_s\":1,\"requests\":1,\"jobs_running\":0,"
                       "\"jobs_capacity\":4,\"ops\":[\"add\",\"relu\"],"
                       "\"max_payload\":1048576}");
        } else {
          sendResponse(fd, 500, kJsonHdr, jsonError("EXEC_FAILED", "boom"));
        }
      },
      /*connections=*/4));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto sleeps = std::make_shared<std::vector<int>>();
  auto res = executeRemoteAsync(s, buildJob(c), recordingOpts(sleeps));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find(proto::kErrUnsupportedOp) != std::string::npos);
  CHECK(res.error.find("matmul") != std::string::npos);
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 1);  // the status pre-flight only
  CHECK(countPrefix(reqs, "POST /api/execute") == 0);
}

int main() {
  test_async_happy_path();
  test_backoff_doubles_and_caps();
  test_poll_timeout_is_bounded();
  test_expired_job_on_poll_is_result_lost();
  test_expired_result_fetch_is_result_lost();
  test_sync_200_answer_to_async_submit_is_protocol_error();
  test_submit_busy_maps_section7_code();
  test_job_error_state_maps_embedded_code();
  test_job_cancelled_state_is_distinct();
  test_undecodable_result_reports_decode_failure();
  test_auto_small_payload_uses_sync();
  test_auto_large_payload_goes_async();
  test_async_preflight_unsupported_op();
  if (g_failures == 0) {
    std::printf("OK: all async jobs tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
