// ============================================================================
// Caramel Language - Device registry + auth sessions (CDP client side)
// ----------------------------------------------------------------------------
// Ticket:  lang_042 (Device Registry + Auth Sessions)
// Version: 1.0.0
// ----------------------------------------------------------------------------
// See include/caramel/net/device.h for the behavioral contract. Secrets
// (passwords, tokens) are deliberately kept out of every message string
// assembled here.
// ============================================================================
#include "caramel/net/device.h"

#include <cstdlib>
#include <utility>

#include "caramel/proto/errors.h"
#include "caramel/proto/json_lite.h"

namespace caramel::net {
namespace {

std::string endpoint(const DeviceConfig &cfg) {
  return cfg.host + ":" + std::to_string(cfg.port);
}

std::string bodyToString(const std::vector<uint8_t> &b) {
  return std::string(b.begin(), b.end());
}

DeviceError transportError(HttpError e, const std::string &what) {
  DeviceError err;
  err.kind = DeviceErrorKind::Transport;
  err.transport = e;
  err.message = what + ": " + httpErrorName(e);
  return err;
}

// Strict decimal port in [1, 65535].
bool parsePort(const std::string &s, uint16_t &out) {
  if (s.empty() || s.size() > 5) return false;
  uint32_t n = 0;
  for (const char c : s) {
    if (c < '0' || c > '9') return false;
    n = n * 10 + static_cast<uint32_t>(c - '0');
  }
  if (n < 1 || n > 65535) return false;
  out = static_cast<uint16_t>(n);
  return true;
}

}  // namespace

const char *deviceErrorKindName(DeviceErrorKind k) {
  switch (k) {
    case DeviceErrorKind::None: return "none";
    case DeviceErrorKind::Transport: return "transport";
    case DeviceErrorKind::AuthFailed: return "auth_failed";
    case DeviceErrorKind::AuthRequired: return "auth_required";
    case DeviceErrorKind::Protocol: return "protocol";
    case DeviceErrorKind::Config: return "config";
  }
  return "unknown";
}

std::string expandEnv(const std::string &value) {
  if (value.rfind("env:", 0) == 0) {
    const char *v = std::getenv(value.c_str() + 4);
    return v ? std::string(v) : std::string();
  }
  return value;
}

DeviceConfigResult DeviceConfig::fromCli(const std::string &url,
                                         const std::string &user,
                                         const std::string &pass,
                                         const std::string &token) {
  DeviceConfigResult result;

  std::string rest = url;
  if (rest.rfind("https://", 0) == 0) {
    result.error =
        "https:// is not supported: CDP v1 is plaintext HTTP on a trusted "
        "LAN (PROTOCOL_SPEC.md Section 5.3); use http://HOST:PORT";
    return result;
  }
  if (rest.rfind("http://", 0) == 0) rest.erase(0, 7);
  if (!rest.empty() && rest.back() == '/') rest.pop_back();  // trailing slash
  if (rest.empty()) {
    result.error = "empty device URL (expected http://HOST:PORT, HOST:PORT, "
                   "or HOST)";
    return result;
  }
  if (rest.find('/') != std::string::npos) {
    result.error = "device URL must not contain a path: \"" + url + "\"";
    return result;
  }

  DeviceConfig cfg;
  const std::size_t colon = rest.find(':');
  if (colon == std::string::npos) {
    cfg.host = rest;
    cfg.port = kDefaultDevicePort;
  } else {
    if (rest.find(':', colon + 1) != std::string::npos) {
      result.error = "invalid device URL (multiple ':'): \"" + url + "\"";
      return result;
    }
    cfg.host = rest.substr(0, colon);
    if (!parsePort(rest.substr(colon + 1), cfg.port)) {
      result.error = "invalid port in device URL: \"" + url + "\"";
      return result;
    }
  }
  if (cfg.host.empty()) {
    result.error = "missing host in device URL: \"" + url + "\"";
    return result;
  }

  // Credential wiring. Secrets may be spelled "env:NAME"; resolve them here
  // (and never echo the resolved values).
  cfg.user = user;
  cfg.pass = expandEnv(pass);
  cfg.token = expandEnv(token);
  if (!cfg.token.empty() && (!user.empty() || !pass.empty())) {
    result.error =
        "specify either --device-user/--device-pass or --device-token, "
        "not both";
    return result;
  }
  if (cfg.token.empty() && !cfg.user.empty() && cfg.pass.empty()) {
    // Pass flag absent but login intended: fall back to the environment so
    // the password does not have to appear in shell history.
    const char *env_pass = std::getenv("CARAMEL_DEVICE_PASS");
    if (env_pass) cfg.pass = env_pass;
  }

  result.config = std::move(cfg);
  return result;
}

DeviceSession::DeviceSession(DeviceConfig cfg, HttpClientOptions opts)
    : cfg_(std::move(cfg)), client_(opts) {
  // Resolve "env:NAME" secrets once, so manually constructed configs behave
  // like fromCli() output. Idempotent for already-literal values.
  cfg_.pass = expandEnv(cfg_.pass);
  cfg_.token = expandEnv(cfg_.token);
  token_ = cfg_.token;  // pre-issued token skips /api/auth entirely
}

DeviceError DeviceSession::login() {
  DeviceError err;
  if (!canLogin()) {
    err.kind = DeviceErrorKind::Config;
    err.message = "no credentials configured for " + endpoint(cfg_) +
                  " (need user+pass or a pre-issued token)";
    return err;
  }

  // PROTOCOL_SPEC.md Section 5.1 body shape, keys in spec order.
  const std::string auth_json = std::string("{\"username\":\"") +
                                proto::jsonEscape(cfg_.user) +
                                "\",\"password\":\"" +
                                proto::jsonEscape(cfg_.pass) + "\"}";
  HttpRequest req;
  req.method = "POST";
  req.host = cfg_.host;
  req.port = cfg_.port;
  req.path = "/api/auth";
  req.headers.emplace_back("Content-Type", "application/json");
  req.body.assign(auth_json.begin(), auth_json.end());

  HttpResult r = client_.request(req);
  if (!r.ok()) {
    return transportError(r.error, "login to " + endpoint(cfg_) + " failed");
  }

  const std::string body = bodyToString(r.response->body);
  if (r.response->status == 200) {
    const auto obj = proto::parseJsonObject(body);
    const auto tok = obj ? obj->getString("token") : std::nullopt;
    if (!tok || tok->empty()) {
      err.kind = DeviceErrorKind::Protocol;
      err.message =
          "malformed /api/auth success body from " + endpoint(cfg_);
      return err;
    }
    // Spec: 32 hex chars. Accepted leniently (any nonempty string) so a
    // future longer token does not break older controllers.
    token_ = *tok;
    return err;  // kind == None
  }

  const auto pe = proto::parseErrorBody(body);
  if (pe) {
    err.proto_code = pe->code;
    err.proto_detail = pe->detail;
  }
  if (r.response->status == 401 && pe && pe->code == proto::kErrAuthFailed) {
    err.kind = DeviceErrorKind::AuthFailed;
    err.message = "authentication failed for " + endpoint(cfg_) + " (" +
                  proto::kErrAuthFailed + ": " + pe->detail + ")";
    return err;
  }
  err.kind = DeviceErrorKind::Protocol;
  err.message = "unexpected /api/auth response " +
                std::to_string(r.response->status) + " from " +
                endpoint(cfg_) + (pe ? " (" + pe->code + ")" : "");
  return err;
}

DeviceResult DeviceSession::authedRequest(const std::string &method,
                                          const std::string &path,
                                          const std::vector<uint8_t> &body,
                                          const std::string &content_type) {
  DeviceResult out;

  if (token_.empty()) {  // lazy login on the first authenticated call
    DeviceError e = login();
    if (e.kind != DeviceErrorKind::None) {
      out.error = std::move(e);
      return out;
    }
  }

  bool relogged = false;
  for (;;) {
    HttpRequest req;
    req.method = method;
    req.host = cfg_.host;
    req.port = cfg_.port;
    req.path = path;
    req.headers.emplace_back("X-Caramel-Token", token_);  // Section 5.2
    if (!content_type.empty()) {
      req.headers.emplace_back("Content-Type", content_type);
    }
    req.body = body;

    HttpResult r = client_.request(req);
    if (!r.ok()) {
      out.error = transportError(
          r.error, method + " " + path + " on " + endpoint(cfg_) + " failed");
      return out;
    }
    if (r.response->status != 401) {
      // Everything else (2xx and non-auth 4xx/5xx) is the caller's to
      // interpret against caramel/proto/errors.h.
      out.response = std::move(*r.response);
      return out;
    }

    const auto pe = proto::parseErrorBody(r.response->body);
    if (!pe) {
      out.error.kind = DeviceErrorKind::Protocol;
      out.error.message =
          "malformed 401 error body from " + endpoint(cfg_) + " for " +
          method + " " + path;
      return out;
    }
    if (pe->code == proto::kErrAuthRequired) {
      // PROTOCOL_SPEC.md Section 5.1 (resolved decision Q1): tokens never
      // expire, but a worker reboot or session-slot eviction invalidates
      // them, surfacing as 401 AUTH_REQUIRED. Client half of that decision:
      // exactly one automatic re-login + retry when user+pass are
      // configured; a second consecutive 401, or a pre-issued token we
      // cannot re-mint, surfaces the error.
      if (!relogged && canLogin()) {
        relogged = true;
        token_.clear();
        DeviceError e = login();
        if (e.kind != DeviceErrorKind::None) {
          out.error = std::move(e);
          return out;
        }
        continue;  // retry the original request once with the fresh token
      }
      out.error.kind = DeviceErrorKind::AuthRequired;
      out.error.proto_code = pe->code;
      out.error.proto_detail = pe->detail;
      out.error.message =
          std::string("token rejected by ") + endpoint(cfg_) + " (" +
          proto::kErrAuthRequired + ": " + pe->detail + ")" +
          (relogged ? " after one re-login"
                    : "; no user/pass configured for re-login");
      return out;
    }
    if (pe->code == proto::kErrAuthFailed) {
      // Bad credentials are never retried (no retry loop on AUTH_FAILED).
      out.error.kind = DeviceErrorKind::AuthFailed;
      out.error.proto_code = pe->code;
      out.error.proto_detail = pe->detail;
      out.error.message = "authentication failed for " + endpoint(cfg_) +
                          " (" + proto::kErrAuthFailed + ": " + pe->detail +
                          ")";
      return out;
    }
    out.error.kind = DeviceErrorKind::Protocol;
    out.error.proto_code = pe->code;
    out.error.proto_detail = pe->detail;
    out.error.message = "unexpected 401 error code \"" + pe->code +
                        "\" from " + endpoint(cfg_);
    return out;
  }
}

DeviceStatusResult DeviceSession::status() {
  DeviceStatusResult out;

  HttpRequest req;
  req.method = "GET";
  req.host = cfg_.host;
  req.port = cfg_.port;
  req.path = "/api/status";  // unauthenticated (Section 6 route table)

  HttpResult r = client_.request(req);
  if (!r.ok()) {
    out.error = transportError(
        r.error, "GET /api/status on " + endpoint(cfg_) + " failed");
    return out;
  }

  const std::string body = bodyToString(r.response->body);
  if (r.response->status != 200) {
    const auto pe = proto::parseErrorBody(body);
    out.error.kind = DeviceErrorKind::Protocol;
    if (pe) {
      out.error.proto_code = pe->code;
      out.error.proto_detail = pe->detail;
      out.error.message = "/api/status on " + endpoint(cfg_) + " returned " +
                          std::to_string(r.response->status) + " (" +
                          pe->code + ")";
    } else {
      out.error.message = "/api/status on " + endpoint(cfg_) + " returned " +
                          std::to_string(r.response->status);
    }
    return out;
  }

  const auto obj = proto::parseJsonObject(body);
  if (!obj) {
    out.error.kind = DeviceErrorKind::Protocol;
    out.error.message = "malformed /api/status JSON from " + endpoint(cfg_);
    return out;
  }

  // Section 6.1 fields; unknown members were already skipped by the parser.
  const auto proto_v = obj->getInt("proto");
  const auto alias = obj->getString("alias");
  const auto kind = obj->getString("kind");
  const auto uptime = obj->getUint("uptime_s");
  const auto requests = obj->getUint("requests");
  const auto jobs_running = obj->getInt("jobs_running");
  const auto jobs_capacity = obj->getInt("jobs_capacity");
  const auto ops = obj->getStringArray("ops");
  const auto max_payload = obj->getUint("max_payload");
  if (!proto_v || !alias || !kind || !uptime || !requests || !jobs_running ||
      !jobs_capacity || !ops || !max_payload) {
    out.error.kind = DeviceErrorKind::Protocol;
    out.error.message =
        "missing or mistyped /api/status field from " + endpoint(cfg_);
    return out;
  }

  DeviceStatus st;
  st.proto = static_cast<int>(*proto_v);
  st.alias = *alias;
  st.kind = *kind;
  st.uptime_s = *uptime;
  st.requests = *requests;
  st.jobs_running = static_cast<int>(*jobs_running);
  st.jobs_capacity = static_cast<int>(*jobs_capacity);
  st.ops = *ops;
  st.max_payload = *max_payload;
  out.status = std::move(st);
  return out;
}

}  // namespace caramel::net
