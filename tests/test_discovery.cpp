// ============================================================================
// Caramel Language - discovery client tests (lang_045)
// ----------------------------------------------------------------------------
// Covers:
//   * parseDiscoveryReply: the x86_045 snapshot fixture, missing/invalid
//     fields, oversized/garbage payloads
//   * discoverDevices against stub UDP workers on 127.0.0.1: reply
//     collection, probe payload validation, multiple workers, dedup of a
//     twice-probed worker, garbage replies skipped, empty result on silence
//   * collectScriptDevices with `discover = true`: alias binding via the
//     source-ip + advertised-port rule, unknown alias diagnostics,
//     host/discover mutual exclusion, port-field rejection
//
// The stub reply fixture below byte-matches the x86_045 host snapshot test
// (tests/discovery_test.c in bark) per that ticket's coordination
// note - the two sides must stay in lockstep on the reply shape.
// ============================================================================
#include "caramel/net/discovery.h"
#include "caramel/net/dispatch.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

using namespace caramel::net;
namespace ast = caramel::ast;
namespace parse = caramel::parse;

static int g_failures = 0;

#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// x86_045 snapshot (idle worker) with a per-test alias substituted in.
static std::string snapshotReply(const std::string &alias) {
  return "{\"proto\":1,\"alias\":\"" + alias +
         "\",\"kind\":\"x86\",\"ip\":\"10.0.2.15\",\"port\":4780,"
         "\"ops\":[\"matmul\",\"add\",\"sub\",\"mul\",\"relu\",\"transpose\"],"
         "\"max_payload\":1048576,\"busy\":false}";
}

// ---------------------------------------------------------------------------
// Stub UDP worker: binds an ephemeral 127.0.0.1 port, answers every valid
// probe with a fixed payload until stopped. Records what it received.
// ---------------------------------------------------------------------------
class StubWorker {
 public:
  explicit StubWorker(std::string reply) : reply_(std::move(reply)) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    ::bind(fd_, (sockaddr *)&a, sizeof a);
    socklen_t len = sizeof a;
    ::getsockname(fd_, (sockaddr *)&a, &len);
    port_ = ntohs(a.sin_port);
    timeval tv{0, 100 * 1000};  // 100 ms poll granularity for stop_
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    thread_ = std::thread([this] { serve(); });
  }

  ~StubWorker() {
    stop_ = true;
    thread_.join();
    ::close(fd_);
  }

  uint16_t port() const { return port_; }
  std::string target() const { return "127.0.0.1:" + std::to_string(port_); }
  int probes_seen() const { return probes_seen_.load(); }
  std::string last_probe() const { return last_probe_; }

 private:
  void serve() {
    char buf[2048];
    while (!stop_) {
      sockaddr_in from{};
      socklen_t from_len = sizeof from;
      ssize_t n = ::recvfrom(fd_, buf, sizeof buf, 0, (sockaddr *)&from,
                             &from_len);
      if (n <= 0) continue;
      last_probe_.assign(buf, (std::size_t)n);
      probes_seen_++;
      if (!reply_.empty()) {
        ::sendto(fd_, reply_.data(), reply_.size(), 0, (const sockaddr *)&from,
                 from_len);
      }
    }
  }

  int fd_ = -1;
  uint16_t port_ = 0;
  std::string reply_;
  std::atomic<bool> stop_{false};
  std::atomic<int> probes_seen_{0};
  std::string last_probe_;
  std::thread thread_;
};

// Loopback-only options: no LAN broadcast from the test suite.
static DiscoveryOptions probeOnly(std::vector<std::string> targets,
                                  int timeout_ms = 400) {
  DiscoveryOptions o;
  o.broadcast = false;
  o.probes = std::move(targets);
  o.timeout_ms = timeout_ms;
  return o;
}

// ---------------------------------------------------------------------------
// (a) reply parsing
// ---------------------------------------------------------------------------
static void test_parse_reply() {
  auto d = parseDiscoveryReply(snapshotReply("caramel-worker"), "192.168.0.42");
  CHECK(d.has_value());
  if (d) {
    CHECK(d->proto == 1);
    CHECK(d->alias == "caramel-worker");
    CHECK(d->kind == "x86");
    CHECK(d->ip == "10.0.2.15");
    CHECK(d->port == 4780);
    CHECK(d->ops.size() == 6 && d->ops[0] == "matmul" &&
          d->ops[5] == "transpose");
    CHECK(d->max_payload == 1048576);
    CHECK(!d->busy);
    CHECK(d->source_ip == "192.168.0.42");
    CHECK(d->connectHost() == "192.168.0.42:4780");
  }

  CHECK(!parseDiscoveryReply("", "1.2.3.4").has_value());
  CHECK(!parseDiscoveryReply("not json at all", "1.2.3.4").has_value());
  CHECK(!parseDiscoveryReply("{\"proto\":1}", "1.2.3.4").has_value());
  // Missing one required field (busy) -> rejected.
  CHECK(!parseDiscoveryReply(
             "{\"proto\":1,\"alias\":\"a\",\"kind\":\"x86\","
             "\"ip\":\"10.0.0.1\",\"port\":4780,\"ops\":[\"matmul\"],"
             "\"max_payload\":1048576}",
             "1.2.3.4")
             .has_value());
  // Invalid port values -> rejected.
  CHECK(!parseDiscoveryReply(snapshotReply("a").replace(
                                 snapshotReply("a").find("4780"), 4, "0"),
                             "1.2.3.4")
             .has_value());
  // Oversized payload -> rejected unparsed.
  std::string big = snapshotReply("a");
  big.append(kDiscoveryReplyMax, ' ');
  CHECK(!parseDiscoveryReply(big, "1.2.3.4").has_value());
}

// ---------------------------------------------------------------------------
// (b) probe/collect against stub workers
// ---------------------------------------------------------------------------
static void test_discover_single() {
  StubWorker w(snapshotReply("bob"));
  auto res = discoverDevices(probeOnly({w.target()}));
  CHECK(res.ok());
  CHECK(res.devices.size() == 1);
  if (res.devices.size() == 1) {
    CHECK(res.devices[0].alias == "bob");
    CHECK(res.devices[0].source_ip == "127.0.0.1");
    CHECK(res.devices[0].connectHost() == "127.0.0.1:4780");
  }
  CHECK(w.probes_seen() == 1);
  CHECK(w.last_probe() == kDiscoveryProbe);  // exact Section 4 payload
}

static void test_discover_multiple_and_dedup() {
  StubWorker bob(snapshotReply("bob"));
  StubWorker alice(snapshotReply("alice"));
  // bob probed twice: the duplicate reply must collapse to one record.
  auto res = discoverDevices(
      probeOnly({bob.target(), alice.target(), bob.target()}));
  CHECK(res.ok());
  CHECK(res.devices.size() == 2);
  bool saw_bob = false, saw_alice = false;
  for (const auto &d : res.devices) {
    if (d.alias == "bob") saw_bob = true;
    if (d.alias == "alice") saw_alice = true;
  }
  CHECK(saw_bob && saw_alice);
}

static void test_discover_garbage_and_silence() {
  StubWorker garbage("{\"totally\":\"unrelated\"}");
  StubWorker mute("");
  StubWorker good(snapshotReply("bob"));
  auto res = discoverDevices(
      probeOnly({garbage.target(), mute.target(), good.target()}));
  CHECK(res.ok());
  CHECK(res.devices.size() == 1);
  if (res.devices.size() == 1) CHECK(res.devices[0].alias == "bob");

  // Nobody at all: ok() with an empty list (not an error).
  auto none = discoverDevices(probeOnly({"127.0.0.1:1"}, 200));
  CHECK(none.ok());
  CHECK(none.devices.empty());

  // Unresolvable explicit probe target: fatal (a typo must not be silent).
  auto bad = discoverDevices(probeOnly({"127.0.0.1:notaport"}));
  CHECK(!bad.ok());
  CHECK(bad.error.find("notaport") == std::string::npos ||
        bad.error.find("invalid port") != std::string::npos);
}

// ---------------------------------------------------------------------------
// (c) collectScriptDevices with discover = true
// ---------------------------------------------------------------------------
struct Parsed {
  std::unique_ptr<ast::Program> program;
  bool ok = false;
};

static Parsed parseSrc(const std::string &src) {
  parse::Lexer lex(src);
  parse::Parser parser(lex.tokenize());
  Parsed p;
  p.program = parser.parse();
  p.ok = parser.ok();
  return p;
}

static const char kPreamble[] =
    "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n";

static void test_collect_with_discovery() {
  StubWorker bob(snapshotReply("bob"));
  auto opts = probeOnly({bob.target()});

  Parsed p = parseSrc(std::string(kPreamble) +
      "device::bob { discover = true; user = \"u\"; pass = env(\"P\"); }\n"
      "calc::lambda_flow f(x) @device(bob) {\n    x x matmul r =\n} return r;\n");
  CHECK(p.ok);
  std::vector<ScriptDevice> devs;
  std::string err;
  CHECK(collectScriptDevices(*p.program, devs, err, &opts));
  CHECK(devs.size() == 1);
  if (devs.size() == 1) {
    CHECK(devs[0].alias == "bob");
    CHECK(devs[0].config.host == "127.0.0.1");
    CHECK(devs[0].config.port == 4780);
  }

  // Alias nobody claims: hard error naming the alias.
  err.clear();
  Parsed q = parseSrc(std::string(kPreamble) +
      "device::carol { discover = true; user = \"u\"; pass = env(\"P\"); }\n");
  CHECK(q.ok);
  CHECK(!collectScriptDevices(*q.program, devs, err, &opts));
  CHECK(err.find("carol") != std::string::npos);
  CHECK(err.find("answered discovery") != std::string::npos);

  // host + discover: mutually exclusive.
  err.clear();
  Parsed r = parseSrc(std::string(kPreamble) +
      "device::bob { host = \"10.0.0.1\"; discover = true; user = \"u\"; "
      "pass = env(\"P\"); }\n");
  CHECK(r.ok);
  CHECK(!collectScriptDevices(*r.program, devs, err, &opts));
  CHECK(err.find("mutually exclusive") != std::string::npos);

  // port + discover: the port comes from the reply.
  err.clear();
  Parsed s = parseSrc(std::string(kPreamble) +
      "device::bob { discover = true; port = 4780; user = \"u\"; "
      "pass = env(\"P\"); }\n");
  CHECK(s.ok);
  CHECK(!collectScriptDevices(*s.program, devs, err, &opts));
  CHECK(err.find("port") != std::string::npos);

  // Bad discover value: diagnosed, not silently ignored.
  err.clear();
  Parsed t = parseSrc(std::string(kPreamble) +
      "device::bob { discover = maybe; user = \"u\"; pass = env(\"P\"); }\n");
  CHECK(t.ok);
  CHECK(!collectScriptDevices(*t.program, devs, err, &opts));
  CHECK(err.find("discover") != std::string::npos);

  // discover = false + host: plain lang_044 behavior, discovery never runs.
  err.clear();
  Parsed u = parseSrc(std::string(kPreamble) +
      "device::bob { discover = false; host = \"10.0.0.1\"; user = \"u\"; "
      "pass = env(\"P\"); }\n");
  CHECK(u.ok);
  CHECK(collectScriptDevices(*u.program, devs, err, &opts));
  CHECK(devs.size() == 1 && devs[0].config.host == "10.0.0.1");
}

// lang_045: hyphenated aliases (spec Section 2's own example is "bob-i3";
// the x86 worker default is "caramel-worker") parse in device:: position and
// in @device(...) and resolve via discovery.
static void test_hyphenated_alias() {
  StubWorker w(snapshotReply("caramel-worker"));
  auto opts = probeOnly({w.target()});

  Parsed p = parseSrc(std::string(kPreamble) +
      "device::caramel-worker { discover = true; user = \"u\"; "
      "pass = env(\"P\"); }\n"
      "calc::lambda_flow f(x) @device(caramel-worker) {\n"
      "    x x matmul r =\n} return r;\n");
  CHECK(p.ok);
  std::vector<ScriptDevice> devs;
  std::string err;
  CHECK(collectScriptDevices(*p.program, devs, err, &opts));
  CHECK(devs.size() == 1 && devs[0].alias == "caramel-worker");

  const ast::LambdaFlow *flow = nullptr;
  for (const auto &it : p.program->items) {
    if (it->kind == ast::NodeKind::LambdaFlow)
      flow = static_cast<const ast::LambdaFlow *>(it.get());
  }
  CHECK(flow != nullptr);
  if (flow) CHECK(deviceAliasOf(*flow) == "caramel-worker");

  // "bob - 3" (spaces) must NOT merge into an alias "bob-3".
  Parsed q = parseSrc(std::string(kPreamble) +
      "device::bob - 3 { host = \"10.0.0.1\"; }\n");
  std::vector<ScriptDevice> qdevs;
  std::string qerr;
  bool qcollected = collectScriptDevices(*q.program, qdevs, qerr, &opts);
  CHECK(!(qcollected && qdevs.size() == 1 && qdevs[0].alias == "bob-3"));
}

int main() {
  test_parse_reply();
  test_discover_single();
  test_discover_multiple_and_dedup();
  test_discover_garbage_and_silence();
  test_collect_with_discovery();
  test_hyphenated_alias();
  if (g_failures == 0) {
    std::printf("test_discovery: all checks passed\n");
    return 0;
  }
  std::printf("test_discovery: %d check(s) FAILED\n", g_failures);
  return 1;
}
