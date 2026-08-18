// ============================================================================
// Caramel Language - Minimal HTTP/1.1 client
// ----------------------------------------------------------------------------
// Ticket:   lang_041 (Minimal HTTP/1.1 Client)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// A small, dependency-free, blocking HTTP/1.1 client over POSIX sockets, per
// PROTOCOL_SPEC.md Sections 3 and 5-7: it is the transport for the CDP JSON
// control plane (/api/auth, /api/status, /api/job) and the binary data plane
// (CRPK request / CRRS result bodies). No TLS, redirects, compression, or
// proxies (out of scope for protocol v1).
//
// Behavior:
//   * One request per HttpClient::request() call; the connection is opened
//     and closed inside the call in v1. Keep-alive/connection reuse can be
//     added later behind the same HttpClient object without API changes.
//   * IPv4 only. `host` may be a dotted-quad numeric IP or a hostname
//     (resolved via getaddrinfo, AF_INET). Note getaddrinfo itself has no
//     timeout; numeric IPs never block.
//   * Always sends, in this order: request line, `Host`, `X-Caramel-Proto: 1`
//     (Section 3), `Content-Length` (only when the body is nonempty), then
//     the caller-supplied headers verbatim. Callers should not re-add those
//     three themselves.
//   * Binary-safe request and response bodies (bytes, never C strings).
//   * Response bodies are delimited by `Content-Length`; without one the
//     client reads to EOF (HTTP/1.0 / `Connection: close` servers).
//     `Transfer-Encoding: chunked` is NOT supported and yields
//     MalformedResponse - CDP workers never send it (spec Section 6 note).
//   * Timeouts: connect timeout (non-blocking connect + poll) and an overall
//     response deadline covering request send through last body byte.
//   * Responses larger than `max_response_bytes` yield TooLarge.
//   * SIGPIPE never terminates the process: SO_NOSIGPIPE on macOS,
//     MSG_NOSIGNAL on Linux.
//
// Error handling follows the repo idiom (proto/crpk.h parsing side): no
// exceptions cross the API; failures are reported through the typed
// HttpError in HttpResult:
//   ConnectFailed     - resolution failure, refused/unreachable connect, or
//                       a socket/send error (peer reset before the response)
//   Timeout           - connect or response deadline expired
//   MalformedResponse - unparsable status line/headers, chunked encoding,
//                       or connection lost mid-response (truncated body)
//   TooLarge          - response exceeded max_response_bytes
// ============================================================================
#ifndef CARAMEL_NET_HTTP_CLIENT_H
#define CARAMEL_NET_HTTP_CLIENT_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace caramel::net {

enum class HttpError {
  None,
  ConnectFailed,
  Timeout,
  MalformedResponse,
  TooLarge,
};

// Stable lowercase name for diagnostics/log lines (e.g. "connect_failed").
const char *httpErrorName(HttpError e);

// One request. `path` is sent verbatim as the request target and must include
// any query string (e.g. "/api/job?id=7"). `headers` are appended after the
// automatic Host / X-Caramel-Proto / Content-Length headers.
struct HttpRequest {
  std::string method = "GET";
  std::string host;      // dotted-quad IPv4 or hostname
  uint16_t port = 80;
  std::string path = "/";
  std::vector<std::pair<std::string, std::string>> headers;
  std::vector<uint8_t> body;
};

// A parsed response. Header names keep the server's spelling; use header()
// for case-insensitive lookup (returns the first match, nullopt if absent).
struct HttpResponse {
  int status = 0;
  std::vector<std::pair<std::string, std::string>> headers;
  std::vector<uint8_t> body;

  std::optional<std::string> header(const std::string &name) const;
};

// Expected-style result: exactly one of `response` (success, error == None)
// or a non-None `error` is set.
struct HttpResult {
  std::optional<HttpResponse> response;
  HttpError error = HttpError::None;

  bool ok() const { return response.has_value(); }
};

struct HttpClientOptions {
  int connect_timeout_ms = 5000;                       // spec default: 5 s
  int response_timeout_ms = 30000;                     // spec default: 30 s
  std::size_t max_response_bytes = 2 * 1024 * 1024;    // 2 MiB cap
};

// Blocking client. Stateless in v1 (a fresh connection per call), but held
// as an object so a keep-alive connection cache can land later without
// touching call sites.
class HttpClient {
 public:
  HttpClient() = default;
  explicit HttpClient(HttpClientOptions opts) : opts_(opts) {}

  // Perform one request. Never throws; see HttpError mapping above.
  HttpResult request(const HttpRequest &req);

  const HttpClientOptions &options() const { return opts_; }

 private:
  HttpClientOptions opts_;
};

// One-shot convenience wrapper around a temporary HttpClient.
HttpResult request(const HttpRequest &req, const HttpClientOptions &opts = {});

}  // namespace caramel::net

#endif  // CARAMEL_NET_HTTP_CLIENT_H
