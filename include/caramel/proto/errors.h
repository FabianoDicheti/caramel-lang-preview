// ============================================================================
// Caramel Language - CDP protocol error codes (PROTOCOL_SPEC.md Section 7)
// ----------------------------------------------------------------------------
// Ticket:   lang_042 (Device Registry + Auth Sessions)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// The single home for the Section 7 error code strings, verbatim from the
// spec, so lang_042/043/044 never scatter string literals. Every non-2xx
// control-plane response carries a JSON body of the shape
//
//   {"error": "AUTH_FAILED", "detail": "bad credentials"}
//
// parseErrorBody() turns that body into a ProtoError (nullopt if the body is
// not such an object). `detail` is optional on the wire; it parses as an
// empty string when absent.
//
// | Code                | HTTP | Meaning                                   |
// |---------------------|------|-------------------------------------------|
// | AUTH_FAILED         | 401  | Bad username/password                     |
// | AUTH_REQUIRED       | 401  | Missing/invalid/expired token             |
// | PROTO_VERSION       | 400  | Unsupported X-Caramel-Proto               |
// | BAD_ENVELOPE        | 400  | Malformed CRPK / inner IR / request body  |
// | UNSUPPORTED_OP      | 422  | IR opcode not in worker's `ops` set       |
// | SHAPE_MISMATCH      | 422  | Tensor dims inconsistent with IR          |
// | TOO_LARGE           | 413  | Payload exceeds `max_payload`             |
// | TOO_LARGE_FOR_SYNC  | 413  | Use async mode                            |
// | BUSY                | 503  | No free job slots                         |
// | NOT_READY           | 409  | Result not yet available                  |
// | NO_SUCH_JOB         | 404  | Unknown/freed job id                      |
// | EXEC_FAILED         | 500  | Runtime failure during execution          |
// ============================================================================
#ifndef CARAMEL_PROTO_ERRORS_H
#define CARAMEL_PROTO_ERRORS_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "caramel/proto/json_lite.h"

namespace caramel::proto {

inline constexpr const char kErrAuthFailed[] = "AUTH_FAILED";
inline constexpr const char kErrAuthRequired[] = "AUTH_REQUIRED";
inline constexpr const char kErrProtoVersion[] = "PROTO_VERSION";
inline constexpr const char kErrBadEnvelope[] = "BAD_ENVELOPE";
inline constexpr const char kErrUnsupportedOp[] = "UNSUPPORTED_OP";
inline constexpr const char kErrShapeMismatch[] = "SHAPE_MISMATCH";
inline constexpr const char kErrTooLarge[] = "TOO_LARGE";
inline constexpr const char kErrTooLargeForSync[] = "TOO_LARGE_FOR_SYNC";
inline constexpr const char kErrBusy[] = "BUSY";
inline constexpr const char kErrNotReady[] = "NOT_READY";
inline constexpr const char kErrNoSuchJob[] = "NO_SUCH_JOB";
inline constexpr const char kErrExecFailed[] = "EXEC_FAILED";

// A parsed Section 7 error body.
struct ProtoError {
  std::string code;    // one of the kErr* strings on a conforming worker
  std::string detail;  // human-readable; empty if the worker sent none
};

// Parse {"error":"...","detail":"..."} (any key order, unknown keys ignored).
// nullopt if the body is not a JSON object with a nonempty string "error".
inline std::optional<ProtoError> parseErrorBody(const std::string &body) {
  const auto obj = parseJsonObject(body);
  if (!obj) return std::nullopt;
  const auto code = obj->getString("error");
  if (!code || code->empty()) return std::nullopt;
  ProtoError e;
  e.code = *code;
  if (const auto d = obj->getString("detail")) e.detail = *d;
  return e;
}

// Convenience overload for raw HTTP response bodies.
inline std::optional<ProtoError> parseErrorBody(
    const std::vector<std::uint8_t> &body) {
  return parseErrorBody(std::string(body.begin(), body.end()));
}

}  // namespace caramel::proto

#endif  // CARAMEL_PROTO_ERRORS_H
