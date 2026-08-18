// ============================================================================
// Caramel Language - UDP discovery client (PROTOCOL_SPEC.md Section 4)
// ----------------------------------------------------------------------------
// Ticket:   lang_045 (Discovery Client)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Controller side of worker discovery: broadcast the "CARAMEL? v1" probe to
// UDP port 4781, collect the unicast JSON replies for a bounded window, and
// parse them into DiscoveredDevice records.
//
// Connect address (documented design decision): the address a discovered
// worker is CONTACTED on is the reply datagram's SOURCE IP plus the
// worker-ADVERTISED port. On a plain LAN the two IPs are identical; when the
// worker sits behind any address translation (QEMU slirp being the everyday
// case) the worker's self-reported "ip" field is unroutable from here while
// the datagram source is exactly the address that reached us. The advertised
// "ip" is still kept on the record for display.
//
// Replies that are malformed, oversized, or missing required Section 4
// fields are skipped silently - a hostile LAN datagram must never break
// discovery. Duplicate replies (same source ip + advertised port, e.g. one
// worker answering both the broadcast and a unicast probe) are de-duplicated
// to the first one seen.
//
// Error handling follows the repo idiom (net/http_client.h): no exceptions
// cross the API; only socket-level setup failures are reported as errors.
// "Nobody answered" is an ok() result with an empty device list.
// ============================================================================
#ifndef CARAMEL_NET_DISCOVERY_H
#define CARAMEL_NET_DISCOVERY_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caramel::net {

// PROTOCOL_SPEC.md Section 3: discovery port; Section 4: probe payload.
inline constexpr uint16_t kDiscoveryPort = 4781;
inline constexpr const char kDiscoveryProbe[] = "CARAMEL? v1";

// Replies larger than this are dropped unparsed (workers budget replies
// under 512 bytes; anything bigger is not one of ours).
inline constexpr std::size_t kDiscoveryReplyMax = 1024;

// One worker's parsed Section 4 reply.
struct DiscoveredDevice {
  int proto = 0;
  std::string alias;
  std::string kind;         // "x86" | "fpga"
  std::string ip;           // worker-advertised (display; see header note)
  uint16_t port = 0;        // worker-advertised API port
  std::vector<std::string> ops;
  uint64_t max_payload = 0;
  bool busy = false;

  std::string source_ip;    // datagram source - the address that reached us
  uint16_t source_port = 0; // datagram source port (dedup key component)

  // "SOURCE_IP:ADVERTISED_PORT" - what a controller should connect to.
  std::string connectHost() const {
    return source_ip + ":" + std::to_string(port);
  }
};

struct DiscoveryOptions {
  int timeout_ms = 1000;              // total listen window
  bool broadcast = true;              // probe 255.255.255.255:port
  std::vector<std::string> probes;    // extra unicast targets "HOST[:PORT]"
  uint16_t port = kDiscoveryPort;     // default target port
};

// Expected-style result: `error` is nonempty only for socket-level setup
// failures; an empty `devices` with an empty `error` means nobody answered.
struct DiscoveryResult {
  std::vector<DiscoveredDevice> devices;
  std::string error;

  bool ok() const { return error.empty(); }
};

// Probe and collect per the header comment. Blocks for ~timeout_ms.
DiscoveryResult discoverDevices(const DiscoveryOptions &opts = {});

// Parse one reply payload (exposed for tests; `source_ip` is copied into the
// record). std::nullopt when malformed or missing a required v1 field.
std::optional<DiscoveredDevice> parseDiscoveryReply(const std::string &json,
                                                    const std::string &source_ip);

}  // namespace caramel::net

#endif  // CARAMEL_NET_DISCOVERY_H
