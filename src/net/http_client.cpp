// ============================================================================
// Caramel Language - Minimal HTTP/1.1 client
// ----------------------------------------------------------------------------
// Ticket:  lang_041 (Minimal HTTP/1.1 Client)
// Version: 1.0.0
// ----------------------------------------------------------------------------
// The socket stays non-blocking for its whole lifetime; every wait (connect,
// send, recv) goes through poll() against either the connect timeout or the
// single response deadline, so no call can block past its budget.
// ============================================================================
#include "caramel/net/http_client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>

namespace caramel::net {
namespace {

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;  // Linux: suppress SIGPIPE per send
#else
constexpr int kSendFlags = 0;             // macOS: SO_NOSIGPIPE on the socket
#endif

constexpr std::size_t kNpos = static_cast<std::size_t>(-1);

char asciiLower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool iequals(const std::string &a, const std::string &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (asciiLower(a[i]) != asciiLower(b[i])) return false;
  }
  return true;
}

std::string lowered(const std::string &s) {
  std::string r = s;
  for (char &c : r) c = asciiLower(c);
  return r;
}

// Trim ASCII SP/HTAB from both ends (header field-value whitespace).
std::string trim(const std::string &s) {
  std::size_t b = 0, e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
  return s.substr(b, e - b);
}

// Absolute deadline; remainingMs() clamps into [0, INT_MAX] for poll().
class Deadline {
 public:
  explicit Deadline(int timeout_ms)
      : end_(std::chrono::steady_clock::now() +
             std::chrono::milliseconds(timeout_ms)) {}

  int remainingMs() const {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_ - std::chrono::steady_clock::now())
                          .count();
    if (left <= 0) return 0;
    if (left > std::numeric_limits<int>::max()) {
      return std::numeric_limits<int>::max();
    }
    return static_cast<int>(left);
  }

 private:
  std::chrono::steady_clock::time_point end_;
};

// RAII fd so every error path closes the socket.
struct Socket {
  int fd = -1;
  ~Socket() {
    if (fd >= 0) ::close(fd);
  }
  Socket() = default;
  Socket(const Socket &) = delete;
  Socket &operator=(const Socket &) = delete;
};

// Fill an IPv4 sockaddr from a dotted-quad literal or via getaddrinfo
// (AF_INET only). Returns false if the host does not resolve to an IPv4
// address. getaddrinfo has no timeout of its own; numeric IPs never block.
bool resolveIPv4(const std::string &host, uint16_t port, sockaddr_in &out) {
  std::memset(&out, 0, sizeof(out));
  out.sin_family = AF_INET;
  out.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &out.sin_addr) == 1) return true;

  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res = nullptr;
  if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) {
    return false;
  }
  bool found = false;
  for (const addrinfo *ai = res; ai; ai = ai->ai_next) {
    if (ai->ai_family == AF_INET && ai->ai_addrlen >= sizeof(sockaddr_in)) {
      out.sin_addr = reinterpret_cast<const sockaddr_in *>(ai->ai_addr)->sin_addr;
      found = true;
      break;
    }
  }
  ::freeaddrinfo(res);
  return found;
}

// Non-blocking connect + poll(POLLOUT) + SO_ERROR check.
HttpError connectWithTimeout(int fd, const sockaddr_in &addr, int timeout_ms) {
  if (::connect(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) == 0) {
    return HttpError::None;  // immediate (loopback fast path)
  }
  if (errno != EINPROGRESS) return HttpError::ConnectFailed;

  Deadline dl(timeout_ms);
  for (;;) {
    const int left = dl.remainingMs();
    if (left <= 0) return HttpError::Timeout;
    pollfd p{};
    p.fd = fd;
    p.events = POLLOUT;
    const int n = ::poll(&p, 1, left);
    if (n < 0) {
      if (errno == EINTR) continue;
      return HttpError::ConnectFailed;
    }
    if (n == 0) return HttpError::Timeout;
    break;
  }
  int soerr = 0;
  socklen_t len = sizeof(soerr);
  if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len) != 0 || soerr != 0) {
    return HttpError::ConnectFailed;
  }
  return HttpError::None;
}

// Send the whole buffer before the deadline.
HttpError sendAll(int fd, const uint8_t *data, std::size_t len, const Deadline &dl) {
  std::size_t off = 0;
  while (off < len) {
    const int left = dl.remainingMs();
    if (left <= 0) return HttpError::Timeout;
    pollfd p{};
    p.fd = fd;
    p.events = POLLOUT;
    const int n = ::poll(&p, 1, left);
    if (n < 0) {
      if (errno == EINTR) continue;
      return HttpError::ConnectFailed;
    }
    if (n == 0) return HttpError::Timeout;
    const ssize_t w = ::send(fd, data + off, len - off, kSendFlags);
    if (w < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return HttpError::ConnectFailed;  // reset/refused before a response
    }
    off += static_cast<std::size_t>(w);
  }
  return HttpError::None;
}

// One recv before the deadline. Returns bytes read (0 = orderly EOF); on
// timeout or socket error returns -1 with `err` set.
ssize_t recvSome(int fd, uint8_t *buf, std::size_t cap, const Deadline &dl,
                 HttpError &err) {
  for (;;) {
    const int left = dl.remainingMs();
    if (left <= 0) {
      err = HttpError::Timeout;
      return -1;
    }
    pollfd p{};
    p.fd = fd;
    p.events = POLLIN;
    const int n = ::poll(&p, 1, left);
    if (n < 0) {
      if (errno == EINTR) continue;
      err = HttpError::MalformedResponse;
      return -1;
    }
    if (n == 0) {
      err = HttpError::Timeout;
      return -1;
    }
    const ssize_t r = ::recv(fd, buf, cap, 0);
    if (r < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      err = HttpError::MalformedResponse;  // connection died mid-response
      return -1;
    }
    return r;
  }
}

// Parse "HTTP/1.x SP+ 3DIGIT [SP reason]" plus the header lines in `head`
// (the bytes before the blank line, CRLF-separated). Returns false on any
// malformed construct; obs-fold continuation lines are rejected.
bool parseHead(const std::string &head, HttpResponse &resp) {
  std::size_t eol = head.find("\r\n");
  const std::string line =
      (eol == std::string::npos) ? head : head.substr(0, eol);

  if (line.size() < 8 || line.compare(0, 7, "HTTP/1.") != 0) return false;
  if (line[7] != '0' && line[7] != '1') return false;
  std::size_t sp = 8;
  if (sp >= line.size() || line[sp] != ' ') return false;
  while (sp < line.size() && line[sp] == ' ') ++sp;
  if (sp + 3 > line.size()) return false;
  int code = 0;
  for (int i = 0; i < 3; ++i) {
    const char c = line[sp + i];
    if (c < '0' || c > '9') return false;
    code = code * 10 + (c - '0');
  }
  if (sp + 3 < line.size() && line[sp + 3] != ' ') return false;
  resp.status = code;

  std::size_t pos = (eol == std::string::npos) ? head.size() : eol + 2;
  while (pos < head.size()) {
    eol = head.find("\r\n", pos);
    const std::string h =
        (eol == std::string::npos) ? head.substr(pos) : head.substr(pos, eol - pos);
    if (!h.empty() && (h[0] == ' ' || h[0] == '\t')) return false;  // obs-fold
    const std::size_t colon = h.find(':');
    if (colon == std::string::npos || colon == 0) return false;
    resp.headers.emplace_back(trim(h.substr(0, colon)), trim(h.substr(colon + 1)));
    if (eol == std::string::npos) break;
    pos = eol + 2;
  }
  return true;
}

// Strict nonnegative decimal (Content-Length grammar).
bool parseContentLength(const std::string &v, uint64_t &out) {
  if (v.empty() || v.size() > 19) return false;
  uint64_t n = 0;
  for (char c : v) {
    if (c < '0' || c > '9') return false;
    n = n * 10 + static_cast<uint64_t>(c - '0');
  }
  out = n;
  return true;
}

}  // namespace

const char *httpErrorName(HttpError e) {
  switch (e) {
    case HttpError::None: return "none";
    case HttpError::ConnectFailed: return "connect_failed";
    case HttpError::Timeout: return "timeout";
    case HttpError::MalformedResponse: return "malformed_response";
    case HttpError::TooLarge: return "too_large";
  }
  return "unknown";
}

std::optional<std::string> HttpResponse::header(const std::string &name) const {
  for (const auto &h : headers) {
    if (iequals(h.first, name)) return h.second;
  }
  return std::nullopt;
}

HttpResult HttpClient::request(const HttpRequest &req) {
  HttpResult result;
  auto fail = [&result](HttpError e) -> HttpResult & {
    result.error = e;
    return result;
  };

  sockaddr_in addr{};
  if (req.method.empty() || req.host.empty() ||
      !resolveIPv4(req.host, req.port, addr)) {
    return fail(HttpError::ConnectFailed);
  }

  Socket sock;
  sock.fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock.fd < 0) return fail(HttpError::ConnectFailed);
  const int fl = ::fcntl(sock.fd, F_GETFL, 0);
  if (fl < 0 || ::fcntl(sock.fd, F_SETFL, fl | O_NONBLOCK) < 0) {
    return fail(HttpError::ConnectFailed);
  }
#ifdef SO_NOSIGPIPE
  {
    int one = 1;
    (void)::setsockopt(sock.fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  }
#endif

  const HttpError cerr = connectWithTimeout(sock.fd, addr, opts_.connect_timeout_ms);
  if (cerr != HttpError::None) return fail(cerr);

  // ---- serialize and send the request (Host / proto / length first) -------
  std::string head;
  head += req.method;
  head += ' ';
  head += req.path.empty() ? "/" : req.path;
  head += " HTTP/1.1\r\nHost: ";
  head += req.host;
  if (req.port != 80) {
    head += ':';
    head += std::to_string(req.port);
  }
  head += "\r\nX-Caramel-Proto: 1\r\n";
  if (!req.body.empty()) {
    head += "Content-Length: " + std::to_string(req.body.size()) + "\r\n";
  }
  for (const auto &h : req.headers) {
    head += h.first;
    head += ": ";
    head += h.second;
    head += "\r\n";
  }
  head += "\r\n";

  const Deadline dl(opts_.response_timeout_ms);  // covers send + full response
  HttpError err = sendAll(sock.fd, reinterpret_cast<const uint8_t *>(head.data()),
                          head.size(), dl);
  if (err == HttpError::None && !req.body.empty()) {
    err = sendAll(sock.fd, req.body.data(), req.body.size(), dl);
  }
  if (err != HttpError::None) return fail(err);

  // ---- read until the CRLFCRLF terminating the response head --------------
  std::vector<uint8_t> raw;
  std::size_t head_end = kNpos;
  uint8_t chunk[4096];
  while (head_end == kNpos) {
    HttpError rerr = HttpError::None;
    const ssize_t n = recvSome(sock.fd, chunk, sizeof(chunk), dl, rerr);
    if (n < 0) return fail(rerr);
    if (n == 0) return fail(HttpError::MalformedResponse);  // EOF before head
    const std::size_t old = raw.size();
    raw.insert(raw.end(), chunk, chunk + n);
    if (raw.size() > opts_.max_response_bytes) return fail(HttpError::TooLarge);
    for (std::size_t i = (old >= 3) ? old - 3 : 0; i + 4 <= raw.size(); ++i) {
      if (raw[i] == '\r' && raw[i + 1] == '\n' && raw[i + 2] == '\r' &&
          raw[i + 3] == '\n') {
        head_end = i;
        break;
      }
    }
  }

  HttpResponse resp;
  if (!parseHead(std::string(raw.begin(),
                             raw.begin() + static_cast<std::ptrdiff_t>(head_end)),
                 resp)) {
    return fail(HttpError::MalformedResponse);
  }

  // Chunked transfer coding is out of scope (workers never send it).
  bool have_cl = false;
  uint64_t content_length = 0;
  for (const auto &h : resp.headers) {
    if (iequals(h.first, "Transfer-Encoding") &&
        lowered(h.second).find("chunked") != std::string::npos) {
      return fail(HttpError::MalformedResponse);
    }
    if (iequals(h.first, "Content-Length")) {
      uint64_t v = 0;
      if (!parseContentLength(h.second, v)) return fail(HttpError::MalformedResponse);
      if (have_cl && v != content_length) return fail(HttpError::MalformedResponse);
      have_cl = true;
      content_length = v;
    }
  }

  // ---- read the body -------------------------------------------------------
  const bool no_body = iequals(req.method, "HEAD") || resp.status / 100 == 1 ||
                       resp.status == 204 || resp.status == 304;
  resp.body.assign(raw.begin() + static_cast<std::ptrdiff_t>(head_end + 4),
                   raw.end());
  if (no_body) {
    resp.body.clear();
  } else if (have_cl) {
    if (content_length > opts_.max_response_bytes) return fail(HttpError::TooLarge);
    if (resp.body.size() > content_length) resp.body.resize(content_length);
    while (resp.body.size() < content_length) {
      HttpError rerr = HttpError::None;
      const ssize_t n = recvSome(sock.fd, chunk, sizeof(chunk), dl, rerr);
      if (n < 0) return fail(rerr);
      if (n == 0) return fail(HttpError::MalformedResponse);  // truncated body
      const std::size_t take =
          std::min(static_cast<std::size_t>(n),
                   static_cast<std::size_t>(content_length) - resp.body.size());
      resp.body.insert(resp.body.end(), chunk, chunk + take);
    }
  } else {
    // No Content-Length: the server delimits the body by closing (HTTP/1.0
    // or Connection: close). Keep-alive CDP workers always send a length.
    for (;;) {
      HttpError rerr = HttpError::None;
      const ssize_t n = recvSome(sock.fd, chunk, sizeof(chunk), dl, rerr);
      if (n < 0) return fail(rerr);
      if (n == 0) break;  // EOF ends the body
      resp.body.insert(resp.body.end(), chunk, chunk + n);
      if (resp.body.size() > opts_.max_response_bytes) return fail(HttpError::TooLarge);
    }
  }

  result.response = std::move(resp);
  return result;
}

HttpResult request(const HttpRequest &req, const HttpClientOptions &opts) {
  return HttpClient(opts).request(req);
}

}  // namespace caramel::net
