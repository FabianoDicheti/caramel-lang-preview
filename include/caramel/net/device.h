// ============================================================================
// Caramel Language - Device registry + auth sessions (CDP client side)
// ----------------------------------------------------------------------------
// Ticket:   lang_042 (Device Registry + Auth Sessions)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Controller-side device management per PROTOCOL_SPEC.md Section 5: hold a
// worker definition (host, port, credentials), log in via POST /api/auth,
// cache the token for the process lifetime, and attach `X-Caramel-Token` to
// authenticated requests through the lang_041 HttpClient.
//
// Behavior:
//   * Lazy login: the first authedRequest() triggers POST /api/auth when no
//     token is cached. A pre-issued token (DeviceConfig::token) skips login
//     entirely.
//   * 401 AUTH_REQUIRED mid-session: per the spec's resolved decision Q1
//     (Section 5.1) tokens never expire, but a worker reboot or session-slot
//     eviction invalidates them. The client half of that decision: exactly
//     ONE automatic re-login + retry (only possible with user+pass); a
//     second consecutive 401, or a pre-issued token we cannot re-mint,
//     surfaces as DeviceErrorKind::AuthRequired.
//   * 401 AUTH_FAILED (bad credentials) is never retried.
//   * Any other HTTP response (2xx, 4xx, 5xx) is returned to the caller as a
//     success at this layer - interpreting BUSY/NOT_READY/... is lang_043's
//     job (codes live in caramel/proto/errors.h).
//   * status(): GET /api/status (unauthenticated, Section 6.1) parsed into
//     DeviceStatus; unknown JSON fields are ignored (forward compatibility).
//
// Secrets:
//   * DeviceConfig::pass / ::token may be written as "env:NAME"; both
//     DeviceConfig::fromCli() and the DeviceSession constructor resolve that
//     via std::getenv (expandEnv()). Additionally, fromCli() falls back to
//     the CARAMEL_DEVICE_PASS environment variable when user+pass login is
//     intended but the pass argument is absent, so passwords need not land
//     in shell history. Resolved secrets are NEVER logged and NEVER appear
//     in any error message produced by this API.
//
// Error handling follows the repo idiom (net/http_client.h): no exceptions
// cross the API; failures are reported through the typed DeviceError:
//   Transport    - HttpClient failed (see DeviceError::transport)
//   AuthFailed   - worker rejected the credentials (AUTH_FAILED); no retry
//   AuthRequired - token rejected (AUTH_REQUIRED) and re-login impossible
//                  or also rejected
//   Protocol     - malformed JSON / unexpected status or error code
//   Config       - unusable configuration (no credentials, bad URL, ...)
// ============================================================================
#ifndef CARAMEL_NET_DEVICE_H
#define CARAMEL_NET_DEVICE_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "caramel/net/http_client.h"

namespace caramel::net {

// PROTOCOL_SPEC.md Section 3: default worker API port.
inline constexpr uint16_t kDefaultDevicePort = 4780;

struct DeviceConfigResult;

// One worker definition. Either user+pass (login flow) OR a pre-issued
// token; `alias` is informational (the worker reports its own alias).
struct DeviceConfig {
  std::string alias;
  std::string host;
  uint16_t port = kDefaultDevicePort;
  std::string user;
  std::string pass;   // may be "env:NAME" (resolved by fromCli / session)
  std::string token;  // pre-issued; may be "env:NAME"

  // Build a config from CLI-style arguments (the caramel-run flag wiring
  // itself is lang_043). `url` accepts "http://HOST:PORT", "HOST:PORT" or
  // "HOST" (default port 4780); "https://" is rejected with a clear error
  // (CDP v1 is plaintext HTTP on a trusted LAN, spec Section 5.3).
  // Secrets: `pass`/`token` go through expandEnv(); when login is intended
  // (token absent) and `pass` is empty, CARAMEL_DEVICE_PASS is consulted.
  static DeviceConfigResult fromCli(const std::string &url,
                                    const std::string &user,
                                    const std::string &pass,
                                    const std::string &token);
};

// Expected-style result for fromCli: exactly one of `config` or a nonempty
// `error` message (which never contains resolved secrets).
struct DeviceConfigResult {
  std::optional<DeviceConfig> config;
  std::string error;

  bool ok() const { return config.has_value(); }
};

// Resolve a secret value: "env:NAME" -> std::getenv("NAME") (empty string if
// the variable is unset); anything else is returned verbatim. A literal
// secret starting with "env:" is therefore not representable (documented
// limitation). The resolved value is never logged.
std::string expandEnv(const std::string &value);

// Parsed GET /api/status body (PROTOCOL_SPEC.md Section 6.1). Unknown JSON
// fields are ignored; all fields below are required in v1.
struct DeviceStatus {
  int proto = 0;
  std::string alias;
  std::string kind;  // "x86" | "fpga"
  uint64_t uptime_s = 0;
  uint64_t requests = 0;
  int jobs_running = 0;
  int jobs_capacity = 0;
  std::vector<std::string> ops;
  uint64_t max_payload = 0;
};

enum class DeviceErrorKind {
  None,
  Transport,
  AuthFailed,
  AuthRequired,
  Protocol,
  Config,
};

// Stable lowercase name for diagnostics/log lines (e.g. "auth_failed").
const char *deviceErrorKindName(DeviceErrorKind k);

struct DeviceError {
  DeviceErrorKind kind = DeviceErrorKind::None;
  HttpError transport = HttpError::None;  // set when kind == Transport
  std::string proto_code;    // Section 7 code, when one was parsed
  std::string proto_detail;  // worker-supplied detail, when present
  std::string message;       // human-readable; never contains secrets
};

// Expected-style result for authedRequest(): exactly one of `response`
// (any HTTP status except auth-handled 401s) or a non-None error.
struct DeviceResult {
  std::optional<HttpResponse> response;
  DeviceError error;

  bool ok() const { return response.has_value(); }
};

// Expected-style result for status().
struct DeviceStatusResult {
  std::optional<DeviceStatus> status;
  DeviceError error;

  bool ok() const { return status.has_value(); }
};

// One authenticated session against one worker. Not thread-safe (one
// session per dispatch thread, matching the single-threaded interpreter).
class DeviceSession {
 public:
  // Resolves "env:NAME" secrets in `cfg` (see expandEnv) and adopts a
  // pre-issued token immediately; no network traffic until the first call.
  explicit DeviceSession(DeviceConfig cfg,
                         HttpClientOptions opts = HttpClientOptions{});

  // Perform one authenticated request (lazy login + 401 handling per the
  // header comment). `content_type` is sent as Content-Type when nonempty;
  // Host / X-Caramel-Proto / Content-Length come from HttpClient.
  DeviceResult authedRequest(const std::string &method,
                             const std::string &path,
                             const std::vector<uint8_t> &body = {},
                             const std::string &content_type = "");

  // GET /api/status (unauthenticated endpoint, Section 6.1).
  DeviceStatusResult status();

  // True once a token is cached (pre-issued or from a successful login).
  bool hasToken() const { return !token_.empty(); }

  const DeviceConfig &config() const { return cfg_; }

 private:
  bool canLogin() const { return !cfg_.user.empty(); }

  // POST /api/auth with the configured user+pass; caches the token on
  // success. Returns a None-kind DeviceError on success.
  DeviceError login();

  DeviceConfig cfg_;
  HttpClient client_;
  std::string token_;  // cached for the process lifetime (spec Section 9)
};

}  // namespace caramel::net

#endif  // CARAMEL_NET_DEVICE_H
