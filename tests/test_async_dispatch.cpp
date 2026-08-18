// ============================================================================
// Caramel Language - Concurrent multi-device dispatch tests (lang_044)
// ----------------------------------------------------------------------------
// Ticket: lang_044 (Async Jobs Client + device:: Syntax in .crml)
// ----------------------------------------------------------------------------
// net/dispatch.h (runFlows) against TWO stub workers (tests/stub_server.h),
// driving the full script surface: device:: blocks parsed from source,
// @device routing, collectScriptDevices, level-synchronous thread-per-flow
// execution:
//   * concurrency: two independent flows on two devices have OVERLAPPING
//     in-flight execute requests (each stub blocks its response until it has
//     seen the other stub's request; serialized dispatch would time the
//     barrier out and fail the test - never hang it)
//   * dependency ordering: a flow consuming another's output runs strictly
//     after it, on a different device, with the CONTROLLER moving the tensor
//     (the consumer's CRPK carries the producer's decoded output; workers
//     never talk to each other)
//   * remote -> local mixing: an undecorated flow runs on the in-process
//     interpreter, fed by a remote flow's output
//   * local-only plans byte-match a direct interpreter run (no network)
//   * actionable errors: unknown @device alias, missing input value
// ============================================================================
#include "caramel/net/dispatch.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "caramel/interp/interpreter.h"
#include "caramel/interp/lambda.h"
#include "caramel/interp/matrix_ops.h"
#include "caramel/ir/dataflow.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"
#include "caramel/proto/crpk.h"
#include "stub_server.h"

using namespace caramel::net;
using namespace caramel::teststub;
namespace ast = caramel::ast;
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

static const char kToken[] = "0123456789abcdef0123456789abcdef";
static const char kJsonHdr[] = "Content-Type: application/json\r\n";
static const char kCrrsHdr[] = "Content-Type: application/x-caramel-crrs\r\n";

// ---------------------------------------------------------------------------
// Helpers: parse a script, build the plan the way caramel-run does
// ---------------------------------------------------------------------------

struct Script {
  std::unique_ptr<ast::Program> program;
  std::vector<PlannedFlow> plan;
  std::vector<ScriptDevice> devices;
};

static Script loadScript(const std::string &src) {
  Script s;
  caramel::parse::Lexer lx(src);
  caramel::parse::Parser p(lx.tokenize());
  s.program = p.parse();
  CHECK(p.ok());
  for (auto &it : s.program->items) {
    if (it->kind != ast::NodeKind::LambdaFlow) continue;
    auto *lf = static_cast<ast::LambdaFlow *>(it.get());
    PlannedFlow pf;
    pf.flow = lf;
    pf.device_alias = deviceAliasOf(*lf);
    s.plan.push_back(pf);
  }
  std::string err;
  CHECK(collectScriptDevices(*s.program, s.devices, err));
  CHECK(err.empty());
  return s;
}

static std::string preambleAndDevices(uint16_t port_alpha,
                                      uint16_t port_beta) {
  return "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
         "device::alpha { host = \"127.0.0.1\"; port = " +
         std::to_string(port_alpha) +
         "; user = \"fabiano\"; pass = env(\"CARAMEL_TEST_DEV_PASS\"); }\n"
         "device::beta  { host = \"127.0.0.1\"; port = " +
         std::to_string(port_beta) +
         "; token = env(\"CARAMEL_TEST_DEV_TOKEN\"); }\n";
}

// A CRRS with one 2x2 int32 output tensor holding `vals`.
static std::string crrsBody(const std::vector<int64_t> &vals) {
  proto::CrrsResponse resp;
  const auto t = proto::tensorFromValue(0, proto::kDTypeInt32,
                                        interp::Value::tensor({2, 2}, vals));
  CHECK(t.has_value());
  resp.outputs.push_back(*t);
  const auto bytes = proto::buildCrrs(resp);
  return std::string(bytes.begin(), bytes.end());
}

// Shared event log so ordering across the two stub threads can be asserted.
struct EventLog {
  std::mutex mu;
  std::vector<std::string> events;

  void add(const std::string &e) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back(e);
  }
  std::vector<std::string> snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    return events;
  }
};

static int indexOf(const std::vector<std::string> &v, const std::string &e) {
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i] == e) return static_cast<int>(i);
  return -1;
}

// A stub worker for one device. On POST /api/execute (any mode) it logs
// "<name>:execute", optionally waits on a two-party barrier, and replies
// with `crrs`. Status/auth are always served.
struct WorkerScript {
  std::string name;
  std::string crrs;
  std::shared_ptr<EventLog> log;
  // Barrier (concurrency proof): when set, the execute handler blocks until
  // `in_flight` reaches 2, up to ~2 s; a timeout sets *overlap_failed
  // instead of hanging the test.
  std::shared_ptr<std::atomic<int>> in_flight;
  std::shared_ptr<std::atomic<bool>> overlap_failed;
};

static StubServer::Handler workerHandler(WorkerScript ws) {
  return [ws](int fd, const std::string &raw) {
    if (raw.rfind("GET /api/status HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, 200, kJsonHdr,
                   "{\"proto\":1,\"alias\":\"" + ws.name +
                       "\",\"kind\":\"x86\",\"uptime_s\":1,\"requests\":1,"
                       "\"jobs_running\":0,\"jobs_capacity\":4,"
                       "\"ops\":[\"matmul\",\"add\",\"sub\",\"mul\",\"relu\","
                       "\"transpose\"],\"max_payload\":1048576}");
    } else if (raw.rfind("POST /api/auth HTTP/1.1\r\n", 0) == 0) {
      sendResponse(fd, 200, kJsonHdr,
                   std::string("{\"token\":\"") + kToken + "\",\"alias\":\"" +
                       ws.name + "\"}");
    } else if (raw.rfind("POST /api/execute", 0) == 0) {
      if (ws.log) ws.log->add(ws.name + ":execute");
      if (ws.in_flight) {
        ws.in_flight->fetch_add(1);
        // Two-party barrier: hold this response until the OTHER worker's
        // execute is also in flight (overlap), or ~2 s pass (dispatch was
        // serialized -> flag, respond anyway, never hang).
        bool overlapped = false;
        for (int i = 0; i < 200; ++i) {
          if (ws.in_flight->load() >= 2) {
            overlapped = true;
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!overlapped) ws.overlap_failed->store(true);
      }
      sendResponse(fd, 200, kCrrsHdr, ws.crrs);
    } else {
      sendResponse(fd, 404, kJsonHdr,
                   "{\"error\":\"NO_SUCH_JOB\",\"detail\":\"bad route\"}");
    }
  };
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
// (a) Concurrency: independent flows on two devices overlap in flight
// ---------------------------------------------------------------------------

static void test_independent_flows_overlap_across_devices() {
  auto log = std::make_shared<EventLog>();
  auto in_flight = std::make_shared<std::atomic<int>>(0);
  auto overlap_failed = std::make_shared<std::atomic<bool>>(false);

  StubServer alpha, beta;
  WorkerScript wa{"alpha", crrsBody({11, 12, 13, 14}), log, in_flight,
                  overlap_failed};
  WorkerScript wb{"beta", crrsBody({21, 22, 23, 24}), log, in_flight,
                  overlap_failed};
  CHECK(alpha.start(workerHandler(wa), /*connections=*/6));
  CHECK(beta.start(workerHandler(wb), /*connections=*/6));

  Script s = loadScript(
      preambleAndDevices(alpha.port(), beta.port()) +
      "calc::lambda_flow f(x, w) @device(alpha) {\n"
      "    x w matmul r1 =\n} return r1;\n"
      "calc::lambda_flow g(y, v) @device(beta) {\n"
      "    y v matmul r2 =\n} return r2;\n");
  CHECK(s.plan.size() == 2);
  CHECK(s.plan[0].device_alias == "alpha");
  CHECK(s.plan[1].device_alias == "beta");

  const std::vector<std::pair<std::string, interp::Value>> inputs = {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
      {"w", interp::Value::tensor({2, 2}, {5, 6, 7, 8})},
      {"y", interp::Value::tensor({2, 2}, {1, 0, 0, 1})},
      {"v", interp::Value::tensor({2, 2}, {2, 0, 0, 2})},
  };
  auto res = runFlows(*s.program, s.plan, s.devices, inputs);
  alpha.stop();
  beta.stop();

  CHECK(res.ok());
  CHECK(res.flows.size() == 2);
  CHECK(res.flows[0].flow_name == "f");
  CHECK(res.flows[0].device_alias == "alpha");
  CHECK(res.flows[0].outputs.size() == 1);
  CHECK(res.flows[0].outputs[0].first == "r1");
  CHECK(res.flows[0].outputs[0].second.data ==
        (std::vector<int64_t>{11, 12, 13, 14}));
  CHECK(res.flows[1].flow_name == "g");
  CHECK(res.flows[1].outputs[0].second.data ==
        (std::vector<int64_t>{21, 22, 23, 24}));

  // The concurrency proof: both executes were in flight simultaneously.
  CHECK(!overlap_failed->load());
  CHECK(executeHits(alpha.requests()) == 1);
  CHECK(executeHits(beta.requests()) == 1);
  const auto events = log->snapshot();
  CHECK(indexOf(events, "alpha:execute") >= 0);
  CHECK(indexOf(events, "beta:execute") >= 0);
}

// ---------------------------------------------------------------------------
// (b) Dependency ordering: consumer runs after producer, controller moves
//     the tensor between devices
// ---------------------------------------------------------------------------

static void test_dependent_flows_serialize_and_move_tensors() {
  auto log = std::make_shared<EventLog>();
  // Sentinel values that the LOCAL pipeline could never produce for these
  // inputs: proof the consumer got the producer's REMOTE result.
  const std::vector<int64_t> sentinel = {101, 102, 103, 104};
  const std::vector<int64_t> final_vals = {999, 998, 997, 996};

  StubServer alpha, beta;
  WorkerScript wa{"alpha", crrsBody(sentinel), log, nullptr, nullptr};
  WorkerScript wb{"beta", crrsBody(final_vals), log, nullptr, nullptr};
  CHECK(alpha.start(workerHandler(wa), /*connections=*/6));
  CHECK(beta.start(workerHandler(wb), /*connections=*/6));

  Script s = loadScript(
      preambleAndDevices(alpha.port(), beta.port()) +
      "calc::lambda_flow prod(x, w) @device(alpha) {\n"
      "    x w matmul mid =\n} return mid;\n"
      "calc::lambda_flow cons(mid, bias) @device(beta) {\n"
      "    mid bias elemwise_add out =\n} return out;\n");
  CHECK(s.plan.size() == 2);

  const std::vector<std::pair<std::string, interp::Value>> inputs = {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
      {"w", interp::Value::tensor({2, 2}, {5, 6, 7, 8})},
      {"bias", interp::Value::tensor({2, 2}, {0, 0, 0, 0})},
      // NOTE: no "mid" here - it must come from prod's remote output.
  };
  auto res = runFlows(*s.program, s.plan, s.devices, inputs);
  alpha.stop();
  beta.stop();

  CHECK(res.ok());
  CHECK(res.flows.size() == 2);
  CHECK(res.flows[0].outputs[0].first == "mid");
  CHECK(res.flows[0].outputs[0].second.data == sentinel);
  CHECK(res.flows[1].outputs[0].first == "out");
  CHECK(res.flows[1].outputs[0].second.data == final_vals);

  // Strict ordering: the consumer's execute happened after the producer's.
  const auto events = log->snapshot();
  const int ia = indexOf(events, "alpha:execute");
  const int ib = indexOf(events, "beta:execute");
  CHECK(ia >= 0 && ib >= 0 && ia < ib);

  // The controller moved the tensor: beta's CRPK input slot 0 ("mid", the
  // first parameter of cons) carries alpha's decoded output, verbatim.
  std::string beta_exec_raw;
  for (const auto &r : beta.requests())
    if (r.rfind("POST /api/execute", 0) == 0) beta_exec_raw = r;
  CHECK(!beta_exec_raw.empty());
  const std::string body = reqBody(beta_exec_raw);
  const auto crpk =
      proto::parseCrpk(std::vector<uint8_t>(body.begin(), body.end()));
  CHECK(crpk.has_value());
  if (crpk) {
    CHECK(crpk->inputs.size() == 2);  // mid + bias
    CHECK(crpk->inputs[0].param_index == 0);
    const auto mid = proto::valueFromTensor(crpk->inputs[0]);
    CHECK(mid.has_value());
    CHECK(mid && mid->data == sentinel);
  }
  // Workers never talk to each other: every request either worker saw came
  // from the controller (status/auth/execute only, one execute each).
  CHECK(executeHits(alpha.requests()) == 1);
  CHECK(executeHits(beta.requests()) == 1);
  for (const auto &r : beta.requests()) {
    CHECK(r.rfind("GET /api/status", 0) == 0 ||
          r.rfind("POST /api/auth", 0) == 0 ||
          r.rfind("POST /api/execute", 0) == 0);
  }
}

// ---------------------------------------------------------------------------
// (c) Remote producer -> local consumer (no @device) mixing
// ---------------------------------------------------------------------------

static void test_remote_feeds_local_flow() {
  auto log = std::make_shared<EventLog>();
  const std::vector<int64_t> sentinel = {50, -60, 70, -80};

  StubServer alpha;
  WorkerScript wa{"alpha", crrsBody(sentinel), log, nullptr, nullptr};
  CHECK(alpha.start(workerHandler(wa), /*connections=*/6));

  // beta's port is unused here; point it at alpha to keep the shared
  // preamble helper (no flow routes to beta).
  Script s = loadScript(
      preambleAndDevices(alpha.port(), alpha.port()) +
      "calc::lambda_flow prod(x, w) @device(alpha) {\n"
      "    x w matmul mid =\n} return mid;\n"
      "calc::lambda_flow post(mid) {\n"
      "    mid relu fin =\n} return fin;\n");
  CHECK(s.plan.size() == 2);
  CHECK(s.plan[1].device_alias.empty());  // undecorated -> local

  const std::vector<std::pair<std::string, interp::Value>> inputs = {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
      {"w", interp::Value::tensor({2, 2}, {5, 6, 7, 8})},
  };
  auto res = runFlows(*s.program, s.plan, s.devices, inputs);
  alpha.stop();

  CHECK(res.ok());
  CHECK(res.flows.size() == 2);
  CHECK(res.flows[1].device_alias.empty());
  CHECK(res.flows[1].outputs[0].first == "fin");
  // relu applied LOCALLY to the remote sentinel.
  CHECK(res.flows[1].outputs[0].second.data ==
        (std::vector<int64_t>{50, 0, 70, 0}));
  CHECK(executeHits(alpha.requests()) == 1);  // only prod went remote
}

// ---------------------------------------------------------------------------
// (d) Local-only plans byte-match a direct interpreter run (no sockets)
// ---------------------------------------------------------------------------

static void test_local_only_plan_matches_interpreter() {
  const std::string src =
      "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
      "calc::lambda_flow layer(x, w, bias) {\n"
      "    x w matmul p =\n"
      "    p bias elemwise_add s =\n"
      "    s relu result =\n"
      "} return result;\n";
  Script s = loadScript(src);
  CHECK(s.devices.empty());
  CHECK(s.plan.size() == 1);
  CHECK(s.plan[0].device_alias.empty());

  const std::vector<std::pair<std::string, interp::Value>> inputs = {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
      {"w", interp::Value::tensor({2, 2}, {5, 6, 7, 8})},
      {"bias", interp::Value::tensor({2, 2}, {0, 0, -100, 0})},
  };
  auto res = runFlows(*s.program, s.plan, s.devices, inputs);
  CHECK(res.ok());
  CHECK(res.flows.size() == 1);

  // Direct interpreter run over the same graph.
  const ast::LambdaFlow *lf = s.plan[0].flow;
  ir::DataflowGraph graph = ir::buildDataflow(*lf);
  interp::Interpreter vm;
  vm.add_evaluator(interp::matrixOpEvaluator());
  vm.add_evaluator(interp::lambdaOpEvaluator());
  for (const auto &kv : inputs) vm.set_input(kv.first, kv.second);
  auto direct = vm.run(graph);
  CHECK(direct.ok());
  const auto it = direct.outputs.find("result");
  CHECK(it != direct.outputs.end());
  CHECK(res.flows[0].outputs.size() == 1);
  CHECK(res.flows[0].outputs[0].second.dims == it->second.dims);
  CHECK(res.flows[0].outputs[0].second.data == it->second.data);
  CHECK(res.flows[0].outputs[0].second.data ==
        (std::vector<int64_t>{19, 22, 0, 50}));
}

// ---------------------------------------------------------------------------
// (e) Actionable errors: unknown alias, missing input
// ---------------------------------------------------------------------------

static void test_unknown_alias_and_missing_input_errors() {
  const std::string src =
      "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
      "device::alpha { host = \"127.0.0.1\"; port = 1; "
      "token = env(\"CARAMEL_TEST_DEV_TOKEN\"); }\n"
      "calc::lambda_flow f(x, w) @device(gamma) {\n"
      "    x w matmul r =\n} return r;\n";
  Script s = loadScript(src);
  const std::vector<std::pair<std::string, interp::Value>> inputs = {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
      {"w", interp::Value::tensor({2, 2}, {5, 6, 7, 8})},
  };
  auto res = runFlows(*s.program, s.plan, s.devices, inputs);
  CHECK(!res.ok());
  CHECK(res.error.find("device::gamma") != std::string::npos);
  CHECK(res.error.find("'f'") != std::string::npos);

  // Missing input: nothing produces "w" and no --in supplies it.
  const std::string src2 =
      "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
      "calc::lambda_flow f(x, w) {\n    x w matmul r =\n} return r;\n";
  Script s2 = loadScript(src2);
  const std::vector<std::pair<std::string, interp::Value>> only_x = {
      {"x", interp::Value::tensor({2, 2}, {1, 2, 3, 4})},
  };
  auto res2 = runFlows(*s2.program, s2.plan, s2.devices, only_x);
  CHECK(!res2.ok());
  CHECK(res2.error.find("'w'") != std::string::npos);
  CHECK(res2.error.find("--in") != std::string::npos);
}

int main() {
  // Script credentials resolve through env() at run time.
  ::setenv("CARAMEL_TEST_DEV_PASS", "sw0rdfish-9", 1);
  ::setenv("CARAMEL_TEST_DEV_TOKEN", kToken, 1);

  test_independent_flows_overlap_across_devices();
  test_dependent_flows_serialize_and_move_tensors();
  test_remote_feeds_local_flow();
  test_local_only_plan_matches_interpreter();
  test_unknown_alias_and_missing_input_errors();
  if (g_failures == 0) {
    std::printf("OK: all async dispatch tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
