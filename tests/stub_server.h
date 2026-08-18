// ============================================================================
// Caramel Language - shared in-process HTTP stub server for tests
// ----------------------------------------------------------------------------
// Ticket: lang_042 (Device Registry + Auth Sessions)
// ----------------------------------------------------------------------------
// Factored out of tests/test_http_client.cpp (lang_041) so test_device.cpp
// can script worker-style auth/status/401 sequences against the same stub.
// Blocking sockets on a dedicated std::thread, bound to 127.0.0.1 on an
// ephemeral port. Mimics the bark worker's HTTP behavior: HTTP/1.1,
// Content-Length-delimited responses, keep-alive (the connection is held
// open until the client closes), and never chunked.
//
// lang_042 additions (behavior for existing lang_041 tests is unchanged):
//   * requests()/requestCount(): a log of every raw request received, so
//     tests can assert exact wire sequences (e.g. auth, req, auth, req).
//   * stop() is interruptible: the accept and keep-alive-drain loops poll a
//     stop flag, so a test can start the server with more connection slots
//     than the scenario should legally use ("assert NO retry happened")
//     without deadlocking stop().
// ============================================================================
#ifndef CARAMEL_TESTS_STUB_SERVER_H
#define CARAMEL_TESTS_STUB_SERVER_H

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caramel::teststub {

#ifdef MSG_NOSIGNAL
inline constexpr int kStubSendFlags = MSG_NOSIGNAL;
#else
inline constexpr int kStubSendFlags = 0;
#endif

inline bool stubSend(int fd, const void *data, size_t len) {
  const char *p = static_cast<const char *>(data);
  size_t off = 0;
  while (off < len) {
    const ssize_t n = ::send(fd, p + off, len - off, kStubSendFlags);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    off += static_cast<size_t>(n);
  }
  return true;
}

inline bool stubSend(int fd, const std::string &s) {
  return stubSend(fd, s.data(), s.size());
}

// Read one full request: head until CRLFCRLF, then Content-Length body bytes.
inline bool stubReadRequest(int fd, std::string &raw) {
  raw.clear();
  char buf[65536];
  size_t head_end = std::string::npos;
  while (head_end == std::string::npos) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) return false;
    raw.append(buf, static_cast<size_t>(n));
    head_end = raw.find("\r\n\r\n");
  }
  std::string head_lc = raw.substr(0, head_end);
  for (char &c : head_lc) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  size_t want = 0;
  const size_t cl = head_lc.find("content-length:");
  if (cl != std::string::npos) {
    want = std::strtoul(head_lc.c_str() + cl + 15, nullptr, 10);
  }
  while (raw.size() - head_end - 4 < want) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) return false;
    raw.append(buf, static_cast<size_t>(n));
  }
  return true;
}

// Serves up to `connections` sequential connections on 127.0.0.1:<ephemeral>.
// Per connection: read one request, invoke the handler (which writes the
// response bytes), then hold the socket open (keep-alive mimicry) until the
// client closes. Handlers that want EOF-delimited bodies call
// shutdown(fd, SHUT_WR) themselves. stop() interrupts a pending accept, so
// serving fewer than `connections` connections is fine.
class StubServer {
 public:
  using Handler = std::function<void(int fd, const std::string &raw_request)>;

  bool start(Handler handler, int connections = 1) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  // ephemeral
    if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
        ::listen(listen_fd_, 4) < 0) {
      return false;
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr *>(&addr), &len) < 0) {
      return false;
    }
    port_ = ntohs(addr.sin_port);

    th_ = std::thread([this, handler, connections] {
      for (int c = 0; c < connections; ++c) {
        // Poll for a pending connection so stop() can interrupt.
        int fd = -1;
        for (;;) {
          if (stop_.load()) return;
          pollfd p{};
          p.fd = listen_fd_;
          p.events = POLLIN;
          const int n = ::poll(&p, 1, 50);
          if (n < 0) {
            if (errno == EINTR) continue;
            return;
          }
          if (n > 0) {
            fd = ::accept(listen_fd_, nullptr, nullptr);
            break;
          }
        }
        if (fd < 0) return;
#ifdef SO_NOSIGPIPE
        int nosig = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosig, sizeof(nosig));
#endif
        std::string raw;
        if (stubReadRequest(fd, raw)) {
          {
            std::lock_guard<std::mutex> lock(mu_);
            last_request_ = raw;
            requests_.push_back(raw);
          }
          handler(fd, raw);
        }
        // Keep-alive: wait for the client to close its end (interruptible
        // via a receive timeout + the stop flag).
        timeval tv{};
        tv.tv_usec = 100000;  // 100 ms
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        char sink;
        for (;;) {
          const ssize_t n = ::recv(fd, &sink, 1, 0);
          if (n > 0) continue;
          if (n == 0) break;  // client closed
          if (errno == EINTR) continue;
          if ((errno == EAGAIN || errno == EWOULDBLOCK) && !stop_.load()) {
            continue;
          }
          break;
        }
        ::close(fd);
      }
    });
    return true;
  }

  void stop() {
    stop_.store(true);
    if (th_.joinable()) th_.join();
    if (listen_fd_ >= 0) {
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
  }

  uint16_t port() const { return port_; }

  std::string lastRequest() {
    std::lock_guard<std::mutex> lock(mu_);
    return last_request_;
  }

  // All raw requests received so far, in arrival order (lang_042).
  std::vector<std::string> requests() {
    std::lock_guard<std::mutex> lock(mu_);
    return requests_;
  }

  size_t requestCount() {
    std::lock_guard<std::mutex> lock(mu_);
    return requests_.size();
  }

  ~StubServer() { stop(); }

 private:
  int listen_fd_ = -1;
  uint16_t port_ = 0;
  std::thread th_;
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::string last_request_;
  std::vector<std::string> requests_;
};

// Worker-style response: HTTP/1.1, Content-Length, connection stays open.
inline void sendResponse(int fd, int status, const std::string &extra_headers,
                         const std::string &body) {
  std::string head = "HTTP/1.1 " + std::to_string(status) + " Whatever\r\n";
  head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  head += extra_headers;
  head += "\r\n";
  stubSend(fd, head);
  stubSend(fd, body);
}

}  // namespace caramel::teststub

#endif  // CARAMEL_TESTS_STUB_SERVER_H
