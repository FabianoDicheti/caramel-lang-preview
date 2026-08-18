// ============================================================================
// Caramel Language - CRPK/CRRS protocol envelopes
// ----------------------------------------------------------------------------
// Ticket:  lang_040 (CRPK/CRRS Codec + Fixture Generator)
// Version: 1.0.0
// ============================================================================
#include "caramel/proto/crpk.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace caramel::proto {

namespace {
constexpr std::array<char, 4> kCrpkMagic = {'C', 'R', 'P', 'K'};
constexpr std::array<char, 4> kCrrsMagic = {'C', 'R', 'R', 'S'};
constexpr uint16_t kEnvelopeVersion = 1;
constexpr std::size_t kCrpkHeaderSize = 24;  // 16-byte header + 8-byte quant
constexpr std::size_t kCrrsHeaderSize = 12;

void put_u16(std::vector<uint8_t> &b, uint16_t v) {
  b.push_back(static_cast<uint8_t>(v & 0xFF));
  b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
void put_u32(std::vector<uint8_t> &b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_i16(std::vector<uint8_t> &b, int16_t v) {
  put_u16(b, static_cast<uint16_t>(v));
}
uint16_t get_u16(const uint8_t *p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}
uint32_t get_u32(const uint8_t *p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
  return v;
}
int16_t get_i16(const uint8_t *p) { return static_cast<int16_t>(get_u16(p)); }

// Element count of a dims vector, or nullopt on zero dims / u64 overflow.
std::optional<uint64_t> elementCount(const std::vector<uint32_t> &dims) {
  uint64_t count = 1;
  for (uint32_t d : dims) {
    if (d == 0) return std::nullopt;
    if (count > std::numeric_limits<uint64_t>::max() / d) return std::nullopt;
    count *= d;
  }
  return count;
}

// Validate an entry for serialization; `expect_index` enforces the dense
// 0-based ordering (input i binds to IR slot i / output i).
void validateEntry(const TensorEntry &t, std::size_t expect_index) {
  if (expect_index > std::numeric_limits<uint8_t>::max() ||
      t.param_index != static_cast<uint8_t>(expect_index)) {
    throw std::invalid_argument("tensor entries must be dense and in slot order");
  }
  const std::size_t elem = dtypeSize(t.dtype);
  if (elem == 0) throw std::invalid_argument("tensor entry has an unknown dtype");
  if (t.dims.size() < kMinRank || t.dims.size() > kMaxRank) {
    throw std::invalid_argument("tensor entry rank must be 1..4");
  }
  const auto count = elementCount(t.dims);
  if (!count) throw std::invalid_argument("tensor entry dims must be nonzero");
  if (*count > std::numeric_limits<uint64_t>::max() / elem ||
      *count * elem != t.data.size()) {
    throw std::invalid_argument("tensor entry data size does not match dims");
  }
}

void appendEntry(std::vector<uint8_t> &out, const TensorEntry &t) {
  out.push_back(t.param_index);
  out.push_back(t.dtype);
  out.push_back(static_cast<uint8_t>(t.dims.size()));
  out.push_back(0);  // reserved
  for (uint32_t d : t.dims) put_u32(out, d);
  out.insert(out.end(), t.data.begin(), t.data.end());
}

// Parse one tensor entry at `off`, advancing it. `expect_index` enforces the
// dense ordering. Every read is bounds-checked against `bytes.size()`.
std::optional<TensorEntry> parseEntry(const std::vector<uint8_t> &bytes,
                                      std::size_t &off, std::size_t expect_index) {
  const std::size_t size = bytes.size();
  auto have = [&](std::size_t n) { return off <= size && n <= size - off; };

  if (!have(4)) return std::nullopt;
  TensorEntry t;
  t.param_index = bytes[off + 0];
  t.dtype = bytes[off + 1];
  const uint8_t rank = bytes[off + 2];
  const uint8_t reserved = bytes[off + 3];
  off += 4;

  if (expect_index > std::numeric_limits<uint8_t>::max() ||
      t.param_index != static_cast<uint8_t>(expect_index)) {
    return std::nullopt;
  }
  const std::size_t elem = dtypeSize(t.dtype);
  if (elem == 0) return std::nullopt;
  if (rank < kMinRank || rank > kMaxRank) return std::nullopt;
  if (reserved != 0) return std::nullopt;

  if (!have(static_cast<std::size_t>(rank) * 4)) return std::nullopt;
  for (uint8_t i = 0; i < rank; ++i) {
    t.dims.push_back(get_u32(&bytes[off]));
    off += 4;
  }
  const auto count = elementCount(t.dims);
  if (!count) return std::nullopt;
  if (*count > std::numeric_limits<uint64_t>::max() / elem) return std::nullopt;
  const uint64_t dataBytes = *count * elem;
  if (dataBytes > std::numeric_limits<std::size_t>::max() ||
      !have(static_cast<std::size_t>(dataBytes))) {
    return std::nullopt;
  }
  t.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(off),
                bytes.begin() + static_cast<std::ptrdiff_t>(off + dataBytes));
  off += static_cast<std::size_t>(dataBytes);
  return t;
}
}  // namespace

std::size_t dtypeSize(uint8_t dtype) {
  if (dtype == kDTypeInt8) return 1;
  if (dtype == kDTypeInt32) return 4;
  return 0;
}

std::vector<uint8_t> serializeCrpk(const CrpkRequest &req) {
  if (req.ir.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("CRPK ir module exceeds the u32 ir_size field");
  }
  if (req.inputs.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("CRPK tensor count exceeds the u32 field");
  }
  for (std::size_t i = 0; i < req.inputs.size(); ++i) validateEntry(req.inputs[i], i);

  std::vector<uint8_t> out;
  out.insert(out.end(), kCrpkMagic.begin(), kCrpkMagic.end());
  put_u16(out, kEnvelopeVersion);
  put_u16(out, /*flags=*/0);
  put_u32(out, static_cast<uint32_t>(req.ir.size()));
  put_u32(out, static_cast<uint32_t>(req.inputs.size()));
  out.push_back(req.quant.quantres);
  out.push_back(req.quant.dtype);
  put_u16(out, /*reserved=*/0);
  put_i16(out, req.quant.quantmin);
  put_i16(out, req.quant.quantmax);
  out.insert(out.end(), req.ir.begin(), req.ir.end());
  for (const auto &t : req.inputs) appendEntry(out, t);
  return out;
}

std::optional<CrpkRequest> parseCrpk(const std::vector<uint8_t> &bytes) {
  if (bytes.size() < kCrpkHeaderSize) return std::nullopt;
  if (std::memcmp(bytes.data(), kCrpkMagic.data(), 4) != 0) return std::nullopt;
  if (get_u16(&bytes[4]) != kEnvelopeVersion) return std::nullopt;
  if (get_u16(&bytes[6]) != 0) return std::nullopt;  // reserved flags

  const uint32_t irSize = get_u32(&bytes[8]);
  const uint32_t tensorCount = get_u32(&bytes[12]);

  CrpkRequest req;
  req.quant.quantres = bytes[16];
  req.quant.dtype = bytes[17];
  if (get_u16(&bytes[18]) != 0) return std::nullopt;  // reserved
  req.quant.quantmin = get_i16(&bytes[20]);
  req.quant.quantmax = get_i16(&bytes[22]);
  if (dtypeSize(req.quant.dtype) == 0) return std::nullopt;

  if (irSize > bytes.size() - kCrpkHeaderSize) return std::nullopt;
  req.ir.assign(bytes.begin() + static_cast<std::ptrdiff_t>(kCrpkHeaderSize),
                bytes.begin() + static_cast<std::ptrdiff_t>(kCrpkHeaderSize + irSize));

  std::size_t off = kCrpkHeaderSize + irSize;
  for (uint32_t i = 0; i < tensorCount; ++i) {
    auto t = parseEntry(bytes, off, i);
    if (!t) return std::nullopt;
    req.inputs.push_back(std::move(*t));
  }
  if (off != bytes.size()) return std::nullopt;  // trailing bytes
  return req;
}

std::vector<uint8_t> buildCrrs(const CrrsResponse &resp) {
  if (resp.outputs.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("CRRS tensor count exceeds the u32 field");
  }
  for (std::size_t i = 0; i < resp.outputs.size(); ++i) validateEntry(resp.outputs[i], i);

  std::vector<uint8_t> out;
  out.insert(out.end(), kCrrsMagic.begin(), kCrrsMagic.end());
  put_u16(out, kEnvelopeVersion);
  put_u16(out, resp.status);
  put_u32(out, static_cast<uint32_t>(resp.outputs.size()));
  for (const auto &t : resp.outputs) appendEntry(out, t);
  return out;
}

std::optional<CrrsResponse> parseCrrs(const std::vector<uint8_t> &bytes) {
  if (bytes.size() < kCrrsHeaderSize) return std::nullopt;
  if (std::memcmp(bytes.data(), kCrrsMagic.data(), 4) != 0) return std::nullopt;
  if (get_u16(&bytes[4]) != kEnvelopeVersion) return std::nullopt;

  CrrsResponse resp;
  resp.status = get_u16(&bytes[6]);
  const uint32_t tensorCount = get_u32(&bytes[8]);

  std::size_t off = kCrrsHeaderSize;
  for (uint32_t i = 0; i < tensorCount; ++i) {
    auto t = parseEntry(bytes, off, i);
    if (!t) return std::nullopt;
    resp.outputs.push_back(std::move(*t));
  }
  if (off != bytes.size()) return std::nullopt;  // trailing bytes
  return resp;
}

std::optional<TensorEntry> tensorFromValue(uint8_t param_index, uint8_t dtype,
                                           const caramel::interp::Value &v) {
  const std::size_t elem = dtypeSize(dtype);
  if (elem == 0) return std::nullopt;
  if (v.dims.size() < kMinRank || v.dims.size() > kMaxRank) return std::nullopt;

  TensorEntry t;
  t.param_index = param_index;
  t.dtype = dtype;
  uint64_t count = 1;
  for (int64_t d : v.dims) {
    if (d <= 0 || d > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return std::nullopt;
    }
    if (count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(d)) {
      return std::nullopt;
    }
    count *= static_cast<uint64_t>(d);
    t.dims.push_back(static_cast<uint32_t>(d));
  }
  if (v.data.size() != count) return std::nullopt;

  const int64_t lo = (dtype == kDTypeInt8)
                         ? std::numeric_limits<int8_t>::min()
                         : std::numeric_limits<int32_t>::min();
  const int64_t hi = (dtype == kDTypeInt8)
                         ? std::numeric_limits<int8_t>::max()
                         : std::numeric_limits<int32_t>::max();
  t.data.reserve(v.data.size() * elem);
  for (int64_t x : v.data) {
    if (x < lo || x > hi) return std::nullopt;
    const auto u = static_cast<uint64_t>(x);
    for (std::size_t i = 0; i < elem; ++i) {
      t.data.push_back(static_cast<uint8_t>((u >> (8 * i)) & 0xFF));
    }
  }
  return t;
}

std::optional<caramel::interp::Value> valueFromTensor(const TensorEntry &t) {
  const std::size_t elem = dtypeSize(t.dtype);
  if (elem == 0) return std::nullopt;
  if (t.dims.size() < kMinRank || t.dims.size() > kMaxRank) return std::nullopt;
  const auto count = elementCount(t.dims);
  if (!count) return std::nullopt;
  if (*count > std::numeric_limits<uint64_t>::max() / elem ||
      *count * elem != t.data.size()) {
    return std::nullopt;
  }

  caramel::interp::Value v;
  for (uint32_t d : t.dims) v.dims.push_back(static_cast<int64_t>(d));
  v.data.reserve(static_cast<std::size_t>(*count));
  for (uint64_t i = 0; i < *count; ++i) {
    const uint8_t *p = &t.data[static_cast<std::size_t>(i) * elem];
    if (t.dtype == kDTypeInt8) {
      v.data.push_back(static_cast<int64_t>(static_cast<int8_t>(p[0])));
    } else {
      v.data.push_back(static_cast<int64_t>(static_cast<int32_t>(get_u32(p))));
    }
  }
  return v;
}

}  // namespace caramel::proto
