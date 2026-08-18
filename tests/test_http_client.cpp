// ============================================================================
// Caramel Language - Minimal HTTP/1.1 client tests
// ----------------------------------------------------------------------------
// Ticket: lang_041 (Minimal HTTP/1.1 Client)
// ----------------------------------------------------------------------------
// All tests run against an in-process stub server (std::thread + loopback
// listening socket; no external processes). The stub mimics the bark
// worker's HTTP behavior: HTTP/1.1, Content-Length-delimited responses,
// keep-alive (the connection is held open until the client closes), and
// never chunked - so the client must finish by Content-Length, not by EOF.
//
// Manual smoke (acceptance criterion "works against python3 -m http.server"):
//   verified 2026-07-17 on macOS (Darwin 25.3.0) against Python 3.13.5:
//   $ cd /tmp/lang041_smoke && python3 -m http.server 8123 --bind 127.0.0.1 &
//   $ ./smoke_get 127.0.0.1 8123 /hello.txt   # tiny driver calling request()
//   -> status 200, Content-Type: text/plain, body "hello from python\n"
//      (18 bytes, matches the Content-Length header)
//   -> GET /missing gave status 404 with the 335-byte HTML error body;
//      host "localhost" (getaddrinfo path) gave the same 200.
//   python answers "HTTP/1.0 200 OK" with Content-Length and closes, which
//   exercises the status-line-variant and close-delimited paths.
// ============================================================================
#include "caramel/net/http_client.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "stub_server.h"  // shared StubServer (factored out in lang_042)

using namespace caramel::net;
using namespace caramel::teststub;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// ---------------------------------------------------------------------------
// Stub server plumbing now lives in tests/stub_server.h (shared with
// test_device.cpp since lang_042): stubSend, stubReadRequest, StubServer,
// sendResponse.
// ---------------------------------------------------------------------------

// Echoes the request body back as application/octet-stream.
static void echoHandler(int fd, const std::string &raw) {
  const size_t head_end = raw.find("\r\n\r\n");
  const std::string body = raw.substr(head_end + 4);
  sendResponse(fd, 200, "Content-Type: application/octet-stream\r\n", body);
}

static HttpRequest makeReq(const std::string &method, uint16_t port,
                           const std::string &path) {
  HttpRequest req;
  req.method = method;
  req.host = "127.0.0.1";
  req.port = port;
  req.path = path;
  return req;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_get_happy_path() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    sendResponse(fd, 200, "Content-Type: application/json\r\n",
                 "{\"proto\":1}");
  }));
  auto res = request(makeReq("GET", srv.port(), "/api/status"));
  srv.stop();
  CHECK(res.ok());
  CHECK(res.error == HttpError::None);
  CHECK(res.response->status == 200);
  const std::string body(res.response->body.begin(), res.response->body.end());
  CHECK(body == "{\"proto\":1}");
  auto ct = res.response->header("Content-Type");
  CHECK(ct.has_value() && *ct == "application/json");
  // Request line + query string went out verbatim.
  CHECK(srv.lastRequest().rfind("GET /api/status HTTP/1.1\r\n", 0) == 0);
}

static void test_post_binary_echo() {
  StubServer srv;
  CHECK(srv.start(echoHandler));
  HttpRequest req = makeReq("POST", srv.port(), "/api/execute?mode=sync");
  req.body = {0x00, 0xFF, 'C', 'R', 'P', 'K', 0x00, 0x7F, 0x80, 0x0A, 0x0D, 0x00};
  auto res = request(req);
  srv.stop();
  CHECK(res.ok());
  CHECK(res.response->status == 200);
  CHECK(res.response->body == req.body);  // NUL/0xFF bytes intact both ways
}

static void test_auto_headers_on_the_wire() {
  StubServer srv;
  CHECK(srv.start(echoHandler));
  HttpRequest req = makeReq("POST", srv.port(), "/api/auth");
  req.body = {'a', 'b', 'c', 'd', 'e'};
  req.headers.emplace_back("X-Caramel-Token", "deadbeef");
  auto res = request(req);
  srv.stop();
  CHECK(res.ok());

  const std::string raw = srv.lastRequest();
  const std::string host_line =
      "Host: 127.0.0.1:" + std::to_string(srv.port()) + "\r\n";
  CHECK(raw.find(host_line) != std::string::npos);
  CHECK(raw.find("X-Caramel-Proto: 1\r\n") != std::string::npos);
  CHECK(raw.find("Content-Length: 5\r\n") != std::string::npos);
  // Caller headers are appended after the automatic ones.
  CHECK(raw.find("X-Caramel-Token: deadbeef\r\n") != std::string::npos);
  CHECK(raw.find("X-Caramel-Proto: 1\r\n") <
        raw.find("X-Caramel-Token: deadbeef\r\n"));
}

static void test_headers_case_insensitive() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    stubSend(fd,
             "HTTP/1.1 200 OK\r\n"
             "cOnTeNt-LeNgTh: 2\r\n"
             "x-CARAMEL-pRoTo: 1\r\n"
             "SERVER:   caramel-stub\t\r\n"
             "\r\nok");
  }));
  auto res = request(makeReq("GET", srv.port(), "/"));
  srv.stop();
  CHECK(res.ok());
  CHECK(res.response->body.size() == 2);  // mixed-case Content-Length honored
  auto proto = res.response->header("X-Caramel-Proto");
  CHECK(proto.has_value() && *proto == "1");
  auto server = res.response->header("server");
  CHECK(server.has_value() && *server == "caramel-stub");  // value trimmed
}

static void test_status_codes() {
  static const int kStatuses[] = {200, 401, 404, 503};
  static std::atomic<int> idx{0};
  StubServer srv;
  CHECK(srv.start(
      [](int fd, const std::string &) {
        const int s = kStatuses[idx++];
        // 401 exercises "no reason phrase"; others get one from sendResponse.
        if (s == 401) {
          stubSend(fd, "HTTP/1.1 401\r\nContent-Length: 0\r\n\r\n");
        } else {
          sendResponse(fd, s, "", "{\"error\":\"X\"}");
        }
      },
      /*connections=*/4));
  for (int expected : kStatuses) {
    auto res = request(makeReq("GET", srv.port(), "/api/job?id=1"));
    CHECK(res.ok());
    CHECK(res.response->status == expected);
  }
  srv.stop();
}

static void test_no_content_length_reads_to_eof() {
  // HTTP/1.0-style server (python3 -m http.server class): no keep-alive,
  // body delimited by connection close.
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    stubSend(fd, "HTTP/1.0 200 OK\r\nConnection: close\r\n\r\neof-delimited");
    ::shutdown(fd, SHUT_WR);  // FIN -> client sees EOF
  }));
  auto res = request(makeReq("GET", srv.port(), "/"));
  srv.stop();
  CHECK(res.ok());
  const std::string body(res.response->body.begin(), res.response->body.end());
  CHECK(body == "eof-delimited");
}

static void test_connect_refused() {
  // Grab an ephemeral port, then close the listener so nothing serves it.
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  CHECK(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  CHECK(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
  socklen_t len = sizeof(addr);
  CHECK(::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0);
  const uint16_t dead_port = ntohs(addr.sin_port);
  ::close(fd);

  auto res = request(makeReq("GET", dead_port, "/"));
  CHECK(!res.ok());
  CHECK(res.error == HttpError::ConnectFailed);
}

static void test_unresolvable_host() {
  HttpRequest req = makeReq("GET", 80, "/");
  req.host = "definitely-not-a-real-host.invalid";
  auto res = request(req);
  CHECK(!res.ok());
  CHECK(res.error == HttpError::ConnectFailed);
}

static void test_response_timeout() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    (void)fd;  // accept + read the request, then never answer
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
  }));
  HttpClientOptions opts;
  opts.response_timeout_ms = 100;  // short so the test stays fast
  const auto t0 = std::chrono::steady_clock::now();
  auto res = request(makeReq("GET", srv.port(), "/"), opts);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
  srv.stop();
  CHECK(!res.ok());
  CHECK(res.error == HttpError::Timeout);
  CHECK(elapsed >= 90 && elapsed < 380);  // deadline honored, not the sleep
}

static void test_garbage_response() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    stubSend(fd, "ZZZZ this is not http\r\n\r\n");
  }));
  auto res = request(makeReq("GET", srv.port(), "/"));
  srv.stop();
  CHECK(!res.ok());
  CHECK(res.error == HttpError::MalformedResponse);
}

static void test_chunked_is_malformed() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    stubSend(fd,
             "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
             "2\r\nok\r\n0\r\n\r\n");
  }));
  auto res = request(makeReq("GET", srv.port(), "/"));
  srv.stop();
  CHECK(!res.ok());
  CHECK(res.error == HttpError::MalformedResponse);
}

static void test_truncated_body_is_malformed() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    stubSend(fd, "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort");
    ::shutdown(fd, SHUT_WR);  // EOF before the promised 100 bytes
  }));
  auto res = request(makeReq("GET", srv.port(), "/"));
  srv.stop();
  CHECK(!res.ok());
  CHECK(res.error == HttpError::MalformedResponse);
}

static void test_response_cap_too_large() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    sendResponse(fd, 200, "", std::string(4096, 'x'));
  }));
  HttpClientOptions opts;
  opts.max_response_bytes = 1024;
  auto res = request(makeReq("GET", srv.port(), "/"), opts);
  srv.stop();
  CHECK(!res.ok());
  CHECK(res.error == HttpError::TooLarge);
}

static void test_one_mib_binary_round_trip() {
  // max_payload-sized transfer (PROTOCOL_SPEC.md: 1048576 bytes).
  StubServer srv;
  CHECK(srv.start(echoHandler));
  HttpRequest req = makeReq("POST", srv.port(), "/api/execute");
  req.body.resize(1024 * 1024);
  for (size_t i = 0; i < req.body.size(); ++i) {
    req.body[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);  // hits 0x00/0xFF
  }
  auto res = request(req);
  srv.stop();
  CHECK(res.ok());
  CHECK(res.response->status == 200);
  CHECK(res.response->body == req.body);  // upload + download byte-identical
}

static void test_hostname_resolution() {
  StubServer srv;
  CHECK(srv.start([](int fd, const std::string &) {
    sendResponse(fd, 200, "", "resolved");
  }));
  HttpRequest req = makeReq("GET", srv.port(), "/");
  req.host = "localhost";  // exercises the getaddrinfo(AF_INET) path
  auto res = request(req);
  srv.stop();
  CHECK(res.ok());
  const std::string body(res.response->body.begin(), res.response->body.end());
  CHECK(body == "resolved");
}

static void test_error_names() {
  CHECK(std::string(httpErrorName(HttpError::None)) == "none");
  CHECK(std::string(httpErrorName(HttpError::ConnectFailed)) == "connect_failed");
  CHECK(std::string(httpErrorName(HttpError::Timeout)) == "timeout");
  CHECK(std::string(httpErrorName(HttpError::MalformedResponse)) ==
        "malformed_response");
  CHECK(std::string(httpErrorName(HttpError::TooLarge)) == "too_large");
}

int main() {
  test_get_happy_path();
  test_post_binary_echo();
  test_auto_headers_on_the_wire();
  test_headers_case_insensitive();
  test_status_codes();
  test_no_content_length_reads_to_eof();
  test_connect_refused();
  test_unresolvable_host();
  test_response_timeout();
  test_garbage_response();
  test_chunked_is_malformed();
  test_truncated_body_is_malformed();
  test_response_cap_too_large();
  test_one_mib_binary_round_trip();
  test_hostname_resolution();
  test_error_names();
  if (g_failures == 0) {
    std::printf("OK: all HTTP client tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
