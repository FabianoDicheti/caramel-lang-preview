// ============================================================================
// Caramel Language - UDP discovery client implementation (lang_045)
// ----------------------------------------------------------------------------
// One UDP socket per discoverDevices() call: send the probe to the broadcast
// address and/or the explicit unicast targets, then poll()-loop on the same
// socket until the window closes, parsing every datagram that arrives. The
// socket is non-blocking for its whole lifetime (same discipline as
// net/http_client.cpp).
// ============================================================================
#include "caramel/net/discovery.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

#include "caramel/proto/json_lite.h"

namespace caramel::net {
namespace {

using Clock = std::chrono::steady_clock;

int elapsedMs(Clock::time_point since) {
  return (int)std::chrono::duration_cast<std::chrono::milliseconds>(
             Clock::now() - since)
      .count();
}

// "HOST[:PORT]" -> sockaddr_in (numeric or resolvable HOST). False on
// failure; discovery treats an unresolvable probe target as fatal (the user
// explicitly asked for it, silence would hide a typo).
bool resolveTarget(const std::string &spec, uint16_t default_port,
                   sockaddr_in &out, std::string &err) {
  std::string host = spec;
  uint16_t port = default_port;
  auto colon = spec.rfind(':');
  if (colon != std::string::npos) {
    host = spec.substr(0, colon);
    const std::string p = spec.substr(colon + 1);
    char *end = nullptr;
    long v = std::strtol(p.c_str(), &end, 10);
    if (p.empty() || *end != '\0' || v <= 0 || v > 65535) {
      err = "invalid port in probe target '" + spec + "'";
      return false;
    }
    port = (uint16_t)v;
  }
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo *res = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) {
    err = "cannot resolve probe target '" + host + "'";
    return false;
  }
  out = *(const sockaddr_in *)res->ai_addr;
  out.sin_port = htons(port);
  freeaddrinfo(res);
  return true;
}

}  // namespace

std::optional<DiscoveredDevice> parseDiscoveryReply(
    const std::string &json, const std::string &source_ip) {
  if (json.empty() || json.size() > kDiscoveryReplyMax) return std::nullopt;
  auto obj = proto::parseJsonObject(json);
  if (!obj) return std::nullopt;

  DiscoveredDevice d;
  // All Section 4 fields are required in v1; a reply missing any of them is
  // not one of ours.
  auto proto_v = obj->getInt("proto");
  auto alias = obj->getString("alias");
  auto kind = obj->getString("kind");
  auto ip = obj->getString("ip");
  auto port = obj->getInt("port");
  auto ops = obj->getStringArray("ops");
  auto max_payload = obj->getInt("max_payload");
  auto busy = obj->getBool("busy");
  if (!proto_v || !alias || alias->empty() || !kind || !ip || !port ||
      *port <= 0 || *port > 65535 || !ops || !max_payload || !busy) {
    return std::nullopt;
  }
  d.proto = (int)*proto_v;
  d.alias = *alias;
  d.kind = *kind;
  d.ip = *ip;
  d.port = (uint16_t)*port;
  d.ops = *ops;
  d.max_payload = (uint64_t)*max_payload;
  d.busy = *busy;
  d.source_ip = source_ip;
  return d;
}

DiscoveryResult discoverDevices(const DiscoveryOptions &opts) {
  DiscoveryResult result;

  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    result.error = "discovery: cannot create UDP socket";
    return result;
  }
  ::fcntl(fd, F_SETFL, O_NONBLOCK);
  int one = 1;
  if (opts.broadcast) {
    ::setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &one, sizeof one);
  }

  // Build the target list up front so a bad --probe fails before any I/O.
  std::vector<sockaddr_in> targets;
  if (opts.broadcast) {
    sockaddr_in b{};
    b.sin_family = AF_INET;
    b.sin_port = htons(opts.port);
    b.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    targets.push_back(b);
  }
  for (const auto &spec : opts.probes) {
    sockaddr_in t{};
    std::string err;
    if (!resolveTarget(spec, opts.port, t, err)) {
      ::close(fd);
      result.error = "discovery: " + err;
      return result;
    }
    targets.push_back(t);
  }
  if (targets.empty()) {
    ::close(fd);
    result.error = "discovery: nothing to probe (broadcast off, no targets)";
    return result;
  }

  const std::size_t probe_len = std::strlen(kDiscoveryProbe);
  for (const auto &t : targets) {
    // Best-effort: an unreachable single target must not abort the sweep
    // (broadcast may be blocked on some interfaces, ICMP refusals arrive as
    // recv errors later and are ignored there).
    (void)::sendto(fd, kDiscoveryProbe, probe_len, 0, (const sockaddr *)&t,
                   sizeof t);
  }

  const auto start = Clock::now();
  char buf[kDiscoveryReplyMax + 1];
  for (;;) {
    int left = opts.timeout_ms - elapsedMs(start);
    if (left <= 0) break;
    pollfd pfd{fd, POLLIN, 0};
    int pr = ::poll(&pfd, 1, left);
    if (pr < 0 && errno == EINTR) continue;
    if (pr <= 0) break;  // window closed or poll failure: stop collecting

    sockaddr_in from{};
    socklen_t from_len = sizeof from;
    ssize_t n = ::recvfrom(fd, buf, sizeof buf, 0, (sockaddr *)&from,
                           &from_len);
    if (n <= 0) continue;                       // ICMP-refusal wakeups etc.
    if ((std::size_t)n > kDiscoveryReplyMax) continue;

    char ipstr[INET_ADDRSTRLEN] = {0};
    ::inet_ntop(AF_INET, &from.sin_addr, ipstr, sizeof ipstr);

    auto dev = parseDiscoveryReply(std::string(buf, (std::size_t)n), ipstr);
    if (!dev) continue;                         // not one of ours: skip
    dev->source_port = ntohs(from.sin_port);

    // Dedup: the same worker answering both the broadcast and a unicast
    // probe replies from the same source ip:port with the same advertised
    // port. Distinct workers sharing an IP (port-forward setups) differ in
    // source port and are kept.
    bool dup = false;
    for (const auto &d : result.devices) {
      if (d.source_ip == dev->source_ip &&
          d.source_port == dev->source_port && d.port == dev->port) {
        dup = true;
        break;
      }
    }
    if (!dup) result.devices.push_back(std::move(*dev));
  }

  ::close(fd);
  return result;
}

}  // namespace caramel::net
