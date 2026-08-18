// ============================================================================
// Caramel Language - Remote execution tests (lang_043, --device pipeline)
// ----------------------------------------------------------------------------
// Ticket: lang_043 (Remote Execution in caramel-run)
// ----------------------------------------------------------------------------
// Full CDP client pipeline against a scripted stub worker (tests/stub_server.h)
// that replays the golden priority/fixtures/matmul_relu.crrs bytes:
//   * happy path: compile matmul_relu -> CRPK -> POST /api/execute?mode=sync
//     -> decoded outputs byte-match the local interpreter run; the CRPK sent
//     on the wire byte-matches the golden matmul_relu.crpk fixture
//   * --verify mechanism: passes against the good stub, detects a corrupted
//     CRRS (flipped value byte, and a broken magic)
//   * pre-flight rejection: reduced `ops`, tiny max_payload, wrong proto -
//     each fails locally and /api/execute is never contacted (request log)
//   * Section 7 error codes from the worker map to distinct one-line messages
//
// Fixture location: $CARAMEL_FIXTURES_DIR if set, else resolved relative to
// this source file (__FILE__/../../priority/fixtures) with cwd-relative
// fallbacks, so the test runs from the repo root, caramel_lang/, or build/.
// ============================================================================
#include "caramel/net/remote_run.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
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
// Golden fixture loading (priority/fixtures, CDP-F01)
// ---------------------------------------------------------------------------

static std::vector<uint8_t> g_crpk;  // matmul_relu.crpk (205 bytes)
static std::vector<uint8_t> g_crrs;  // matmul_relu.crrs (40 bytes)

static std::string dirOf(const std::string &p) {
  const auto s = p.find_last_of('/');
  return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

static bool readFile(const std::string &path, std::vector<uint8_t> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.assign(std::istreambuf_iterator<char>(f),
             std::istreambuf_iterator<char>());
  return true;
}

// Resolve priority/fixtures robustly: env override first, then relative to
// this source file (works for standalone and CMake builds, which both pass
// the source path through), then cwd-relative fallbacks.
static std::string fixturesDir() {
  std::vector<std::string> candidates;
  if (const char *env = std::getenv("CARAMEL_FIXTURES_DIR")) {
    candidates.push_back(env);
  }
  candidates.push_back(dirOf(__FILE__) + "/../../priority/fixtures");
  candidates.push_back("../priority/fixtures");    // cwd = caramel_lang/
  candidates.push_back("../../priority/fixtures"); // cwd = caramel_lang/build/
  candidates.push_back("priority/fixtures");       // cwd = repo parent
  for (const auto &c : candidates) {
    std::vector<uint8_t> probe;
    if (readFile(c + "/matmul_relu.crrs", probe)) return c;
  }
  return std::string();
}

// ---------------------------------------------------------------------------
// The canonical flow, compiled exactly as caramel-run does (crpk-gen inputs)
// ---------------------------------------------------------------------------

// Verbatim examples/matmul_relu.crml (minus comments); the fixture source.
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

// Canonical fixture inputs (fixtures_spec.md / crpk-gen defaults), in flow
// parameter declaration order.
static std::vector<std::pair<std::string, interp::Value>> canonicalInputs() {
  return {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
      {"w", interp::Value::tensor({2, 2}, {5, 6, 7, 8})},
      {"bias", interp::Value::tensor({2, 2}, {0, 0, -100, 0})},
  };
}

static RemoteJob buildJob(const Compiled &c) {
  RemoteJob job;
  job.quant.quantres = 0;  // crml::quant* directives of the fixture flow
  job.quant.dtype = proto::kDTypeInt32;
  job.quant.quantmin = -1000;
  job.quant.quantmax = 1000;
  job.ir = ir::serialize(c.graph);
  job.inputs = canonicalInputs();
  job.output_names = c.returns;
  return job;
}

static interp::RunResult runLocal(const Compiled &c) {
  interp::Interpreter vm;
  vm.add_evaluator(interp::matrixOpEvaluator());
  vm.add_evaluator(interp::lambdaOpEvaluator());
  for (const auto &kv : canonicalInputs()) vm.set_input(kv.first, kv.second);
  return vm.run(c.graph);
}

// ---------------------------------------------------------------------------
// Stub worker (spec Sections 5, 6.1, 6.2, 7)
// ---------------------------------------------------------------------------

static const char kToken[] = "0123456789abcdef0123456789abcdef";
static const char kJsonHdr[] = "Content-Type: application/json\r\n";
static const char kDefaultOpsJson[] =
    "[\"matmul\",\"add\",\"sub\",\"mul\",\"relu\",\"transpose\"]";

struct WorkerOptions {
  int proto = 1;
  std::string ops_json = kDefaultOpsJson;
  uint64_t max_payload = 1048576;
  int exec_status = 200;
  std::string exec_content_type = "application/x-caramel-crrs";
  std::string exec_body;  // CRRS bytes (200) or a Section 7 JSON error
};

static std::string statusJson(const WorkerOptions &opt) {
  return "{\"proto\":" + std::to_string(opt.proto) +
         ",\"alias\":\"stub\",\"kind\":\"x86\",\"uptime_s\":1,"
         "\"requests\":1,\"jobs_running\":0,\"jobs_capacity\":4,"
         "\"ops\":" + opt.ops_json +
         ",\"max_payload\":" + std::to_string(opt.max_payload) + "}";
}

static std::string jsonError(const std::string &code, const std::string &detail) {
  return "{\"error\":\"" + code + "\",\"detail\":\"" + detail + "\"}";
}

static StubServer::Handler workerHandler(WorkerOptions opt) {
  return [opt](int fd, const std::string &raw) {
    if (raw.rfind("GET /api/status HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, 200, kJsonHdr, statusJson(opt));
    } else if (raw.rfind("POST /api/auth HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, 200, kJsonHdr,
                   std::string("{\"token\":\"") + kToken +
                       "\",\"alias\":\"stub\"}");
    } else if (raw.rfind("POST /api/execute?mode=sync HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, opt.exec_status,
                   "Content-Type: " + opt.exec_content_type + "\r\n",
                   opt.exec_body);
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

static std::string reqBody(const std::string &raw) {
  const size_t p = raw.find("\r\n\r\n");
  return (p == std::string::npos) ? std::string() : raw.substr(p + 4);
}

static size_t executeHits(const std::vector<std::string> &reqs) {
  size_t n = 0;
  for (const auto &r : reqs)
    if (r.rfind("POST /api/execute", 0) == 0) ++n;
  return n;
}

// ---------------------------------------------------------------------------
// Mnemonic / requiredOps helpers
// ---------------------------------------------------------------------------

static void test_worker_mnemonics_and_required_ops() {
  CHECK(std::string(workerMnemonic(ir::OpCode::MATMUL)) == "matmul");
  CHECK(std::string(workerMnemonic(ir::OpCode::ADD)) == "add");
  CHECK(std::string(workerMnemonic(ir::OpCode::RELU)) == "relu");
  CHECK(std::string(workerMnemonic(ir::OpCode::TRANSPOSE)) == "transpose");
  CHECK(std::string(workerMnemonic(ir::OpCode::CONV2D)) == "conv2d");
  // Structural opcodes are not capability-gated (spec 6.1 ops are compute).
  CHECK(workerMnemonic(ir::OpCode::PARAM) == nullptr);
  CHECK(workerMnemonic(ir::OpCode::SYNC) == nullptr);
  CHECK(workerMnemonic(ir::OpCode::RETURN) == nullptr);
  CHECK(workerMnemonic(ir::OpCode::CONST) == nullptr);

  Compiled c = compileLayer();
  const auto ops = requiredOps(ir::serialize(c.graph));
  CHECK(ops.has_value());
  const std::vector<std::string> want = {"matmul", "add", "relu"};
  CHECK(*ops == want);  // distinct compute mnemonics, first-use order

  const std::vector<uint8_t> garbage = {'n', 'o', 't', ' ', 'i', 'r'};
  CHECK(!requiredOps(garbage).has_value());
}

// ---------------------------------------------------------------------------
// Error mapping: one distinct, actionable line per Section 7 code
// ---------------------------------------------------------------------------

static void test_error_message_mapping() {
  const std::vector<std::string> codes = {
      proto::kErrAuthFailed,   proto::kErrAuthRequired,
      proto::kErrProtoVersion, proto::kErrBadEnvelope,
      proto::kErrUnsupportedOp, proto::kErrShapeMismatch,
      proto::kErrTooLarge,     proto::kErrTooLargeForSync,
      proto::kErrBusy,         proto::kErrNotReady,
      proto::kErrNoSuchJob,    proto::kErrExecFailed,
  };
  std::set<std::string> distinct;
  for (const auto &code : codes) {
    const std::string msg = remoteErrorMessage(code, "");
    CHECK(!msg.empty());
    CHECK(msg.find(code) != std::string::npos);  // names its code
    CHECK(msg.find('\n') == std::string::npos);  // one line
    distinct.insert(msg);
  }
  CHECK(distinct.size() == codes.size());  // pairwise distinct

  // Worker-supplied detail is appended; unknown codes get a generic line.
  CHECK(remoteErrorMessage(proto::kErrBusy, "queue full")
            .find("queue full") != std::string::npos);
  CHECK(remoteErrorMessage("FUTURE_CODE", "")
            .find("FUTURE_CODE") != std::string::npos);
}

// ---------------------------------------------------------------------------
// (a) Happy path: golden CRRS replay == local interpreter run
// ---------------------------------------------------------------------------

static void test_happy_path_matches_local() {
  WorkerOptions opt;
  opt.exec_body.assign(g_crrs.begin(), g_crrs.end());
  StubServer srv;
  CHECK(srv.start(workerHandler(opt), /*connections=*/3));

  Compiled c = compileLayer();
  RemoteJob job = buildJob(c);
  DeviceSession s = makeSession(srv.port());
  auto res = executeRemoteSync(s, job);
  srv.stop();

  CHECK(res.ok());
  CHECK(res.error.empty());
  CHECK(res.outputs->size() == 1);
  CHECK((*res.outputs)[0].first == "result");
  const interp::Value &v = (*res.outputs)[0].second;
  CHECK(v.dims == (std::vector<int64_t>{2, 2}));
  CHECK(v.data == (std::vector<int64_t>{19, 22, 0, 50}));

  // Byte-exact agreement with the local interpreter (M1 criterion).
  auto local = runLocal(c);
  CHECK(local.ok());
  CHECK(verifyAgainstLocal(*res.outputs, local.outputs).empty());

  // Wire sequence: pre-flight status, lazy login, then exactly one execute
  // carrying the token, the CRPK content type, and the golden request bytes.
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 3);
  CHECK(reqs[0].rfind("GET /api/status HTTP/1.1\r\n", 0) == 0);
  CHECK(reqs[1].rfind("POST /api/auth HTTP/1.1\r\n", 0) == 0);
  CHECK(reqs[2].rfind("POST /api/execute?mode=sync HTTP/1.1\r\n", 0) == 0);
  CHECK(reqs[2].find(std::string("X-Caramel-Token: ") + kToken + "\r\n") !=
        std::string::npos);
  CHECK(reqs[2].find("Content-Type: application/x-caramel-crpk\r\n") !=
        std::string::npos);
  CHECK(reqBody(reqs[2]) == std::string(g_crpk.begin(), g_crpk.end()));
}

// ---------------------------------------------------------------------------
// (b) --verify mechanism: PASS on the good stub, MISMATCH on corrupted CRRS
// ---------------------------------------------------------------------------

static void test_verify_detects_corrupted_values() {
  // Flip the first output element (i32 LSB at offset 12 header + 4 entry
  // header + 8 dims = 24): 19 -> 108. Still a well-formed CRRS.
  WorkerOptions opt;
  opt.exec_body.assign(g_crrs.begin(), g_crrs.end());
  opt.exec_body[24] = static_cast<char>(opt.exec_body[24] ^ 0x7F);
  StubServer srv;
  CHECK(srv.start(workerHandler(opt), /*connections=*/3));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto res = executeRemoteSync(s, buildJob(c));
  srv.stop();

  CHECK(res.ok());  // decode succeeds; only the values are wrong
  auto local = runLocal(c);
  CHECK(local.ok());
  const std::string mismatch = verifyAgainstLocal(*res.outputs, local.outputs);
  CHECK(!mismatch.empty());
  CHECK(mismatch.find("result") != std::string::npos);
  CHECK(mismatch.find("differs") != std::string::npos);
}

static void test_corrupted_crrs_magic_is_rejected() {
  WorkerOptions opt;
  opt.exec_body.assign(g_crrs.begin(), g_crrs.end());
  opt.exec_body[0] = 'X';  // break the "CRRS" magic
  StubServer srv;
  CHECK(srv.start(workerHandler(opt), /*connections=*/3));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto res = executeRemoteSync(s, buildJob(c));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("malformed CRRS") != std::string::npos);
}

// ---------------------------------------------------------------------------
// (c) Pre-flight rejection: /api/execute is never contacted
// ---------------------------------------------------------------------------

static void test_preflight_unsupported_op() {
  WorkerOptions opt;
  opt.ops_json = "[\"add\",\"relu\"]";  // no matmul
  opt.exec_body.assign(g_crrs.begin(), g_crrs.end());
  StubServer srv;
  // Extra connection slots: an (incorrect) execute would land in the log.
  CHECK(srv.start(workerHandler(opt), /*connections=*/3));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto res = executeRemoteSync(s, buildJob(c));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find(proto::kErrUnsupportedOp) != std::string::npos);
  CHECK(res.error.find("matmul") != std::string::npos);  // names missing ops
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 1);  // status only: no auth, no execute round trip
  CHECK(reqs[0].rfind("GET /api/status HTTP/1.1\r\n", 0) == 0);
  CHECK(executeHits(reqs) == 0);
}

static void test_preflight_too_large() {
  WorkerOptions opt;
  opt.max_payload = 16;  // smaller than any CRPK (header alone is 24 bytes)
  opt.exec_body.assign(g_crrs.begin(), g_crrs.end());
  StubServer srv;
  CHECK(srv.start(workerHandler(opt), /*connections=*/3));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto res = executeRemoteSync(s, buildJob(c));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find(proto::kErrTooLarge) != std::string::npos);
  CHECK(res.error.find("16") != std::string::npos);  // quotes the limit
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 1);
  CHECK(reqs[0].rfind("GET /api/status HTTP/1.1\r\n", 0) == 0);
  CHECK(executeHits(reqs) == 0);
}

static void test_preflight_proto_mismatch() {
  WorkerOptions opt;
  opt.proto = 2;
  opt.exec_body.assign(g_crrs.begin(), g_crrs.end());
  StubServer srv;
  CHECK(srv.start(workerHandler(opt), /*connections=*/3));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto res = executeRemoteSync(s, buildJob(c));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find(proto::kErrProtoVersion) != std::string::npos);
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 1);
  CHECK(reqs[0].rfind("GET /api/status HTTP/1.1\r\n", 0) == 0);
  CHECK(executeHits(reqs) == 0);
}

// ---------------------------------------------------------------------------
// (d) Worker error codes surface as their distinct mapped messages
// ---------------------------------------------------------------------------

static void test_worker_error_codes() {
  struct Case {
    const char *code;
    int http;
    const char *detail;
  };
  const Case cases[] = {
      {proto::kErrBusy, 503, "no free slots"},
      {proto::kErrExecFailed, 500, "i8 overflow at instr 4"},
      {proto::kErrBadEnvelope, 400, "bad magic"},
      {proto::kErrShapeMismatch, 422, "param 1 dims"},
      {proto::kErrTooLargeForSync, 413, "use async"},
  };
  std::set<std::string> distinct;
  for (const auto &cse : cases) {
    WorkerOptions opt;
    opt.exec_status = cse.http;
    opt.exec_content_type = "application/json";
    opt.exec_body = jsonError(cse.code, cse.detail);
    StubServer srv;
    CHECK(srv.start(workerHandler(opt), /*connections=*/3));

    Compiled c = compileLayer();
    DeviceSession s = makeSession(srv.port());
    auto res = executeRemoteSync(s, buildJob(c));
    srv.stop();

    CHECK(!res.ok());
    CHECK(res.error == remoteErrorMessage(cse.code, cse.detail));
    CHECK(res.error.find(cse.code) != std::string::npos);
    CHECK(res.error.find(cse.detail) != std::string::npos);
    CHECK(executeHits(srv.requests()) == 1);  // pre-flight passed, one POST
    distinct.insert(res.error);
  }
  CHECK(distinct.size() == sizeof(cases) / sizeof(cases[0]));
}

static void test_async_202_is_rejected_in_sync_mode() {
  WorkerOptions opt;
  opt.exec_status = 202;
  opt.exec_content_type = "application/json";
  opt.exec_body = "{\"job\":7,\"state\":\"queued\"}";
  StubServer srv;
  CHECK(srv.start(workerHandler(opt), /*connections=*/3));

  Compiled c = compileLayer();
  DeviceSession s = makeSession(srv.port());
  auto res = executeRemoteSync(s, buildJob(c));
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.find("async") != std::string::npos);  // lang_044 territory
}

int main() {
  const std::string fixtures = fixturesDir();
  if (fixtures.empty()) {
    std::printf("FAIL cannot locate priority/fixtures "
                "(set CARAMEL_FIXTURES_DIR)\n");
    return EXIT_FAILURE;
  }
  CHECK(readFile(fixtures + "/matmul_relu.crpk", g_crpk));
  CHECK(readFile(fixtures + "/matmul_relu.crrs", g_crrs));
  CHECK(g_crpk.size() == 205);  // fixtures_spec.md golden sizes
  CHECK(g_crrs.size() == 40);
  const auto golden = proto::parseCrrs(g_crrs);
  CHECK(golden.has_value());
  CHECK(golden->status == 0);
  if (g_failures != 0) {
    std::printf("FAILED: %d check(s) while loading fixtures\n", g_failures);
    return EXIT_FAILURE;
  }

  test_worker_mnemonics_and_required_ops();
  test_error_message_mapping();
  test_happy_path_matches_local();
  test_verify_detects_corrupted_values();
  test_corrupted_crrs_magic_is_rejected();
  test_preflight_unsupported_op();
  test_preflight_too_large();
  test_preflight_proto_mismatch();
  test_worker_error_codes();
  test_async_202_is_rejected_in_sync_mode();
  if (g_failures == 0) {
    std::printf("OK: all remote execution tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
