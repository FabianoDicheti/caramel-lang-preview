// ============================================================================
// Caramel Language - Device registry + auth session tests
// ----------------------------------------------------------------------------
// Ticket: lang_042 (Device Registry + Auth Sessions)
// ----------------------------------------------------------------------------
// Scripted stub-server sequences (tests/stub_server.h, the lang_041 pattern)
// asserting the exact wire behavior of PROTOCOL_SPEC.md Section 5 client
// side: lazy login, X-Caramel-Token attachment, AUTH_FAILED without retry,
// AUTH_REQUIRED eviction with exactly one re-login, pre-issued tokens that
// never touch /api/auth, /api/status parsing with unknown-field tolerance,
// fromCli URL parsing, and the secrets policy (env expansion; passwords and
// tokens never appear in error messages).
// ============================================================================
#include "caramel/net/device.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "caramel/proto/errors.h"
#include "caramel/proto/json_lite.h"
#include "stub_server.h"

using namespace caramel::net;
using namespace caramel::teststub;
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
// Helpers
// ---------------------------------------------------------------------------

static const char kTokenA[] = "0123456789abcdef0123456789abcdef";  // 32 hex
static const char kTokenB[] = "fedcba9876543210fedcba9876543210";  // 32 hex
static const char kUser[] = "fabiano";
static const char kPass[] = "sw0rdfish-9";

static DeviceConfig cfgUserPass(uint16_t port) {
  DeviceConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = port;
  cfg.user = kUser;
  cfg.pass = kPass;
  return cfg;
}

static std::string reqBody(const std::string &raw) {
  const size_t p = raw.find("\r\n\r\n");
  return (p == std::string::npos) ? std::string() : raw.substr(p + 4);
}

static bool isAuthPost(const std::string &raw) {
  return raw.rfind("POST /api/auth HTTP/1.1\r\n", 0) == 0;
}

static bool hasTokenHeader(const std::string &raw, const std::string &token) {
  return raw.find("X-Caramel-Token: " + token + "\r\n") != std::string::npos;
}

static std::string jsonError(const char *code, const char *detail) {
  return std::string("{\"error\":\"") + code + "\",\"detail\":\"" + detail +
         "\"}";
}

static std::string jsonAuthOk(const char *token) {
  return std::string("{\"token\":\"") + token + "\",\"alias\":\"stub\"}";
}

static const char kJsonHdr[] = "Content-Type: application/json\r\n";

// ---------------------------------------------------------------------------
// Login + token attachment
// ---------------------------------------------------------------------------

static void test_login_happy_path() {
  StubServer srv;
  std::atomic<int> conn{0};
  CHECK(srv.start(
      [&](int fd, const std::string &) {
        if (conn++ == 0) {
          sendResponse(fd, 200, kJsonHdr, jsonAuthOk(kTokenA));
        } else {
          sendResponse(fd, 202, kJsonHdr, "{\"job\":7,\"state\":\"queued\"}");
        }
      },
      /*connections=*/3));

  DeviceSession s(cfgUserPass(srv.port()));
  CHECK(!s.hasToken());  // lazy: no traffic yet

  const std::vector<uint8_t> payload = {'C', 'R', 'P', 'K'};
  auto res = s.authedRequest("POST", "/api/execute", payload,
                             "application/x-caramel-crpk");
  CHECK(res.ok());
  CHECK(res.response->status == 202);
  CHECK(s.hasToken());

  // A second call reuses the cached token: no further /api/auth.
  auto res2 = s.authedRequest("GET", "/api/job?id=7");
  CHECK(res2.ok());
  srv.stop();

  const auto reqs = srv.requests();
  CHECK(reqs.size() == 3);
  // Wire order + exact spec body on the login (Section 5.1).
  CHECK(isAuthPost(reqs[0]));
  CHECK(reqBody(reqs[0]) ==
        std::string("{\"username\":\"") + kUser + "\",\"password\":\"" +
            kPass + "\"}");
  CHECK(reqs[0].find(kJsonHdr) != std::string::npos);
  // The authed requests carry the raw X-Caramel-Token header (Section 5.2).
  CHECK(reqs[1].rfind("POST /api/execute HTTP/1.1\r\n", 0) == 0);
  CHECK(hasTokenHeader(reqs[1], kTokenA));
  CHECK(reqs[1].find("Content-Type: application/x-caramel-crpk\r\n") !=
        std::string::npos);
  CHECK(reqBody(reqs[1]) == "CRPK");
  CHECK(reqs[2].rfind("GET /api/job?id=7 HTTP/1.1\r\n", 0) == 0);
  CHECK(hasTokenHeader(reqs[2], kTokenA));
  CHECK(!isAuthPost(reqs[2]));
}

static void test_wrong_password_no_retry() {
  StubServer srv;
  // Two connection slots: an (incorrect) retry would land in the log.
  CHECK(srv.start(
      [](int fd, const std::string &) {
        sendResponse(fd, 401, kJsonHdr,
                     jsonError(proto::kErrAuthFailed, "bad credentials"));
      },
      /*connections=*/2));

  DeviceConfig cfg = cfgUserPass(srv.port());
  DeviceSession s(cfg);
  auto res = s.authedRequest("GET", "/api/job?id=1");
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.kind == DeviceErrorKind::AuthFailed);
  CHECK(res.error.proto_code == proto::kErrAuthFailed);
  CHECK(res.error.proto_detail == "bad credentials");
  CHECK(!s.hasToken());
  // Exactly one /api/auth hit; bad credentials are never retried.
  CHECK(srv.requestCount() == 1);
  CHECK(isAuthPost(srv.requests()[0]));
  // Secrets policy: the password never appears in the error surface.
  CHECK(!res.error.message.empty());
  CHECK(res.error.message.find(kPass) == std::string::npos);
}

// ---------------------------------------------------------------------------
// Eviction (401 AUTH_REQUIRED) handling - spec Section 5.1, decision Q1
// ---------------------------------------------------------------------------

static void test_eviction_relogin_once() {
  StubServer srv;
  std::atomic<int> conn{0};
  CHECK(srv.start(
      [&](int fd, const std::string &raw) {
        switch (conn++) {
          case 0:  // lazy login
            sendResponse(fd, 200, kJsonHdr, jsonAuthOk(kTokenA));
            break;
          case 1:  // token evicted (worker rebooted / slot reused)
            CHECK(hasTokenHeader(raw, kTokenA));
            sendResponse(fd, 401, kJsonHdr,
                         jsonError(proto::kErrAuthRequired, "evicted"));
            break;
          case 2:  // automatic re-login
            sendResponse(fd, 200, kJsonHdr, jsonAuthOk(kTokenB));
            break;
          default:  // retried request with the fresh token
            CHECK(hasTokenHeader(raw, kTokenB));
            sendResponse(fd, 200, kJsonHdr, "{\"job\":1,\"state\":\"done\"}");
            break;
        }
      },
      /*connections=*/4));

  DeviceSession s(cfgUserPass(srv.port()));
  auto res = s.authedRequest("GET", "/api/job?id=1");
  srv.stop();

  CHECK(res.ok());
  CHECK(res.response->status == 200);

  // Exact sequence on the wire: auth, req(401), auth, req(200).
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 4);
  CHECK(isAuthPost(reqs[0]));
  CHECK(reqs[1].rfind("GET /api/job?id=1 HTTP/1.1\r\n", 0) == 0);
  CHECK(isAuthPost(reqs[2]));
  CHECK(reqs[3].rfind("GET /api/job?id=1 HTTP/1.1\r\n", 0) == 0);
}

static void test_eviction_second_401_surfaces() {
  StubServer srv;
  std::atomic<int> conn{0};
  // Extra connection slots would expose an over-eager retry loop.
  CHECK(srv.start(
      [&](int fd, const std::string &) {
        switch (conn++) {
          case 0:
            sendResponse(fd, 200, kJsonHdr, jsonAuthOk(kTokenA));
            break;
          case 2:  // re-login is accepted...
            sendResponse(fd, 200, kJsonHdr, jsonAuthOk(kTokenB));
            break;
          default:  // ...but the protected route keeps saying 401
            sendResponse(fd, 401, kJsonHdr,
                         jsonError(proto::kErrAuthRequired, "evicted"));
            break;
        }
      },
      /*connections=*/6));

  DeviceSession s(cfgUserPass(srv.port()));
  auto res = s.authedRequest("GET", "/api/result?id=1");
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.kind == DeviceErrorKind::AuthRequired);
  CHECK(res.error.proto_code == proto::kErrAuthRequired);

  // auth, req(401), auth, req(401) -> surfaced. Exactly 2 auth calls total.
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 4);
  int auth_calls = 0;
  for (const auto &r : reqs) {
    if (isAuthPost(r)) ++auth_calls;
  }
  CHECK(auth_calls == 2);
}

// ---------------------------------------------------------------------------
// Pre-issued token
// ---------------------------------------------------------------------------

static void test_preissued_token_skips_login() {
  StubServer srv;
  CHECK(srv.start(
      [](int fd, const std::string &raw) {
        CHECK(!isAuthPost(raw));  // stub asserts: never /api/auth
        sendResponse(fd, 200, kJsonHdr, "{\"job\":3,\"state\":\"running\"}");
      },
      /*connections=*/2));

  DeviceConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = srv.port();
  cfg.token = kTokenA;
  DeviceSession s(cfg);
  CHECK(s.hasToken());  // adopted at construction, no traffic

  auto res = s.authedRequest("GET", "/api/job?id=3");
  srv.stop();

  CHECK(res.ok());
  const auto reqs = srv.requests();
  CHECK(reqs.size() == 1);
  CHECK(reqs[0].rfind("GET /api/job?id=3 HTTP/1.1\r\n", 0) == 0);
  CHECK(hasTokenHeader(reqs[0], kTokenA));
}

static void test_preissued_token_401_surfaces_without_retry() {
  StubServer srv;
  CHECK(srv.start(
      [](int fd, const std::string &) {
        sendResponse(fd, 401, kJsonHdr,
                     jsonError(proto::kErrAuthRequired, "evicted"));
      },
      /*connections=*/2));

  DeviceConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = srv.port();
  cfg.token = kTokenB;
  DeviceSession s(cfg);
  auto res = s.authedRequest("DELETE", "/api/job?id=9");
  srv.stop();

  CHECK(!res.ok());
  CHECK(res.error.kind == DeviceErrorKind::AuthRequired);
  // Token-only config: no re-login possible, no retry, no /api/auth.
  CHECK(srv.requestCount() == 1);
  CHECK(!isAuthPost(srv.requests()[0]));
  // Secrets policy: the token never appears in the error surface.
  CHECK(res.error.message.find(kTokenB) == std::string::npos);
}

// ---------------------------------------------------------------------------
// GET /api/status parsing (Section 6.1)
// ---------------------------------------------------------------------------

static void test_status_parse_full() {
  // Verbatim Section 6.1 example (whitespace + newlines included).
  static const char kStatusJson[] =
      "{\n"
      "  \"proto\": 1,\n"
      "  \"alias\": \"bob-i3\",\n"
      "  \"kind\": \"x86\",\n"
      "  \"uptime_s\": 812,\n"
      "  \"requests\": 1042,\n"
      "  \"jobs_running\": 1,\n"
      "  \"jobs_capacity\": 4,\n"
      "  \"ops\": [\"matmul\",\"add\",\"sub\",\"mul\",\"relu\",\"transpose\"],\n"
      "  \"max_payload\": 1048576\n"
      "}";
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    sendResponse(fd, 200, kJsonHdr, kStatusJson);
  }));

  DeviceConfig cfg;  // no credentials: /api/status is unauthenticated
  cfg.host = "127.0.0.1";
  cfg.port = srv.port();
  DeviceSession s(cfg);
  auto res = s.status();
  srv.stop();

  CHECK(res.ok());
  CHECK(res.status->proto == 1);
  CHECK(res.status->alias == "bob-i3");
  CHECK(res.status->kind == "x86");
  CHECK(res.status->uptime_s == 812);
  CHECK(res.status->requests == 1042);
  CHECK(res.status->jobs_running == 1);
  CHECK(res.status->jobs_capacity == 4);
  const std::vector<std::string> want_ops = {"matmul", "add",  "sub",
                                             "mul",    "relu", "transpose"};
  CHECK(res.status->ops == want_ops);
  CHECK(res.status->max_payload == 1048576);
  // Unauthenticated GET, no token header.
  CHECK(srv.lastRequest().rfind("GET /api/status HTTP/1.1\r\n", 0) == 0);
  CHECK(srv.lastRequest().find("X-Caramel-Token") == std::string::npos);
}

static void test_status_unknown_fields_ignored() {
  // Forward compatibility: extra string, float, mixed array, and nested
  // object members from a future worker must be skipped, any key order.
  static const char kStatusJson[] =
      "{ \"future_str\": \"v2-feature\", \"proto\": 1,"
      " \"future_num\": 3.25, \"alias\": \"bob-i3\", \"kind\": \"fpga\","
      " \"future_arr\": [1, {\"a\": 2}, [3, \"x\"], null],"
      " \"uptime_s\": 5, \"requests\": 6, \"jobs_running\": 0,"
      " \"future_obj\": {\"nested\": {\"deep\": true, \"list\": [1,2]}},"
      " \"jobs_capacity\": 2, \"ops\": [\"matmul\"], \"max_payload\": 4096,"
      " \"busy\": false }";
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    sendResponse(fd, 200, kJsonHdr, kStatusJson);
  }));

  DeviceConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = srv.port();
  DeviceSession s(cfg);
  auto res = s.status();
  srv.stop();

  CHECK(res.ok());
  CHECK(res.status->proto == 1);
  CHECK(res.status->alias == "bob-i3");
  CHECK(res.status->kind == "fpga");
  CHECK(res.status->uptime_s == 5);
  CHECK(res.status->requests == 6);
  CHECK(res.status->jobs_running == 0);
  CHECK(res.status->jobs_capacity == 2);
  CHECK(res.status->ops == std::vector<std::string>{"matmul"});
  CHECK(res.status->max_payload == 4096);
}

// ---------------------------------------------------------------------------
// DeviceConfig::fromCli URL parsing
// ---------------------------------------------------------------------------

static void test_from_cli_url_parsing() {
  auto r1 = DeviceConfig::fromCli("http://bob:1234", "u", "p", "");
  CHECK(r1.ok());
  CHECK(r1.config->host == "bob");
  CHECK(r1.config->port == 1234);
  CHECK(r1.config->user == "u");
  CHECK(r1.config->pass == "p");

  auto r2 = DeviceConfig::fromCli("bob:1234", "", "", kTokenA);
  CHECK(r2.ok());
  CHECK(r2.config->host == "bob");
  CHECK(r2.config->port == 1234);
  CHECK(r2.config->token == kTokenA);

  auto r3 = DeviceConfig::fromCli("bob", "u", "p", "");
  CHECK(r3.ok());
  CHECK(r3.config->host == "bob");
  CHECK(r3.config->port == kDefaultDevicePort);  // 4780 (spec Section 3)

  auto r4 = DeviceConfig::fromCli("http://192.168.0.42:4780/", "u", "p", "");
  CHECK(r4.ok());  // trailing slash tolerated
  CHECK(r4.config->host == "192.168.0.42");
  CHECK(r4.config->port == 4780);

  auto r5 = DeviceConfig::fromCli("https://bob:1234", "u", "p", "");
  CHECK(!r5.ok());
  CHECK(r5.error.find("https") != std::string::npos);  // clear rejection

  CHECK(!DeviceConfig::fromCli("", "u", "p", "").ok());
  CHECK(!DeviceConfig::fromCli("bob:99999", "u", "p", "").ok());
  CHECK(!DeviceConfig::fromCli("bob:0", "u", "p", "").ok());
  CHECK(!DeviceConfig::fromCli("bob:12:34", "u", "p", "").ok());
  CHECK(!DeviceConfig::fromCli("bob:abc", "u", "p", "").ok());
  CHECK(!DeviceConfig::fromCli("http://bob/api", "u", "p", "").ok());
  CHECK(!DeviceConfig::fromCli(":4780", "u", "p", "").ok());
  // user+pass and token are mutually exclusive.
  CHECK(!DeviceConfig::fromCli("bob", "u", "p", kTokenA).ok());
}

// ---------------------------------------------------------------------------
// Secrets: env expansion + never-echoed guarantees
// ---------------------------------------------------------------------------

static void test_env_expansion() {
  ::setenv("CARAMEL_TEST_PASS", "hunter2", 1);
  CHECK(expandEnv("env:CARAMEL_TEST_PASS") == "hunter2");
  CHECK(expandEnv("plain-value") == "plain-value");
  ::unsetenv("CARAMEL_TEST_MISSING");
  CHECK(expandEnv("env:CARAMEL_TEST_MISSING").empty());

  // fromCli resolves env: forms for pass and token.
  auto r1 = DeviceConfig::fromCli("bob", "u", "env:CARAMEL_TEST_PASS", "");
  CHECK(r1.ok());
  CHECK(r1.config->pass == "hunter2");

  ::setenv("CARAMEL_TEST_TOKEN", kTokenA, 1);
  auto r2 = DeviceConfig::fromCli("bob", "", "", "env:CARAMEL_TEST_TOKEN");
  CHECK(r2.ok());
  CHECK(r2.config->token == kTokenA);

  // Pass flag absent + login intended -> CARAMEL_DEVICE_PASS fallback.
  ::setenv("CARAMEL_DEVICE_PASS", "fallback-pw", 1);
  auto r3 = DeviceConfig::fromCli("bob", "u", "", "");
  CHECK(r3.ok());
  CHECK(r3.config->pass == "fallback-pw");
  ::unsetenv("CARAMEL_DEVICE_PASS");
  auto r4 = DeviceConfig::fromCli("bob", "u", "", "");
  CHECK(r4.ok());
  CHECK(r4.config->pass.empty());

  ::unsetenv("CARAMEL_TEST_PASS");
  ::unsetenv("CARAMEL_TEST_TOKEN");
}

static void test_session_expands_env_secrets() {
  // A manually built config (not via fromCli) with an env: password must be
  // resolved by the session before it reaches the wire.
  ::setenv("CARAMEL_TEST_PASS", "hunter2", 1);
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    sendResponse(fd, 401, kJsonHdr,
                 jsonError(proto::kErrAuthFailed, "bad credentials"));
  }));

  DeviceConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = srv.port();
  cfg.user = "u";
  cfg.pass = "env:CARAMEL_TEST_PASS";
  DeviceSession s(cfg);
  auto res = s.authedRequest("GET", "/api/job?id=1");
  srv.stop();

  CHECK(reqBody(srv.requests()[0]) ==
        "{\"username\":\"u\",\"password\":\"hunter2\"}");
  CHECK(!res.ok());
  // Neither the resolved secret nor its env: spelling leaks into errors.
  CHECK(res.error.message.find("hunter2") == std::string::npos);
  ::unsetenv("CARAMEL_TEST_PASS");
}

// ---------------------------------------------------------------------------
// proto/errors.h: Section 7 strings + error body parsing
// ---------------------------------------------------------------------------

static void test_error_codes_and_body_parsing() {
  // Verbatim Section 7 strings.
  CHECK(std::string(proto::kErrAuthFailed) == "AUTH_FAILED");
  CHECK(std::string(proto::kErrAuthRequired) == "AUTH_REQUIRED");
  CHECK(std::string(proto::kErrProtoVersion) == "PROTO_VERSION");
  CHECK(std::string(proto::kErrBadEnvelope) == "BAD_ENVELOPE");
  CHECK(std::string(proto::kErrUnsupportedOp) == "UNSUPPORTED_OP");
  CHECK(std::string(proto::kErrShapeMismatch) == "SHAPE_MISMATCH");
  CHECK(std::string(proto::kErrTooLarge) == "TOO_LARGE");
  CHECK(std::string(proto::kErrTooLargeForSync) == "TOO_LARGE_FOR_SYNC");
  CHECK(std::string(proto::kErrBusy) == "BUSY");
  CHECK(std::string(proto::kErrNotReady) == "NOT_READY");
  CHECK(std::string(proto::kErrNoSuchJob) == "NO_SUCH_JOB");
  CHECK(std::string(proto::kErrExecFailed) == "EXEC_FAILED");

  // Whitespace, key order, and unknown keys tolerated.
  auto e1 = proto::parseErrorBody(std::string(
      " { \"detail\" : \"no free slots\" , \"error\" : \"BUSY\", "
      "\"hint\": 42 } "));
  CHECK(e1.has_value());
  CHECK(e1->code == proto::kErrBusy);
  CHECK(e1->detail == "no free slots");

  // detail is optional on the wire.
  auto e2 = proto::parseErrorBody(std::string("{\"error\":\"NOT_READY\"}"));
  CHECK(e2.has_value());
  CHECK(e2->code == proto::kErrNotReady);
  CHECK(e2->detail.empty());

  CHECK(!proto::parseErrorBody(std::string("not json")).has_value());
  CHECK(!proto::parseErrorBody(std::string("{\"detail\":\"x\"}")).has_value());
  CHECK(!proto::parseErrorBody(std::string("{\"error\":42}")).has_value());
}

int main() {
  test_login_happy_path();
  test_wrong_password_no_retry();
  test_eviction_relogin_once();
  test_eviction_second_401_surfaces();
  test_preissued_token_skips_login();
  test_preissued_token_401_surfaces_without_retry();
  test_status_parse_full();
  test_status_unknown_fields_ignored();
  test_from_cli_url_parsing();
  test_env_expansion();
  test_session_expands_env_secrets();
  test_error_codes_and_body_parsing();
  if (g_failures == 0) {
    std::printf("OK: all device session tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
