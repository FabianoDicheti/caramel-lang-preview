// ============================================================================
// Caramel Language - CRPK/CRRS protocol envelopes
// ----------------------------------------------------------------------------
// Ticket:   lang_040 (CRPK/CRRS Codec + Fixture Generator)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Pure (no networking) codec for the Caramel Distribution Protocol binary
// envelopes, per PROTOCOL_SPEC.md Sections 6.5-6.6:
//   * CRPK - execute request: header, quant block, embedded binary IR module
//     (verbatim caramel::ir::serialize() output, treated as opaque bytes), and
//     the input tensor table.
//   * CRRS - result envelope: header, status, and the output tensor table.
//
// All integers are little-endian and are read/written byte-wise (never via
// struct casts) so the bytes match the freestanding C parser on the worker.
//
// CRPK layout:
//   Offset  Size        Field
//   0       4           magic "CRPK"
//   4       2           version (=1)
//   6       2           flags (reserved, 0)
//   8       4           ir_size
//   12      4           tensor_count
//   16      8           quant block: quantres u8, dtype u8 (1=int8, 2=int32),
//                       reserved u16 (0), quantmin i16, quantmax i16
//   24      ir_size     IR module (opaque, "CRML" ABI v1)
//   ...                 tensor_count tensor entries
//
// Tensor entry layout (shared by CRPK inputs and CRRS outputs):
//   Offset  Size        Field
//   0       1           param_index (IR input slot / output index)
//   1       1           dtype (1=int8, 2=int32)
//   2       1           rank (1..4)
//   3       1           reserved (0)
//   4       4*rank      dims (u32 each)
//   ...     elem*count  data, row-major little-endian
//
// CRRS layout:
//   Offset  Size        Field
//   0       4           magic "CRRS"
//   4       2           version (=1)
//   6       2           status (0 = ok)
//   8       4           tensor_count
//   12      ...         tensor entries (param_index = output index)
//
// Error handling mirrors ir/binary_ir.h: serialization throws on requests that
// cannot be encoded; parsing returns std::nullopt on any malformed input
// (bounds-checked, overflow-safe, no UB).
// ============================================================================
#ifndef CARAMEL_PROTO_CRPK_H
#define CARAMEL_PROTO_CRPK_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "caramel/interp/value.h"

namespace caramel::proto {

// Wire dtype codes (PROTOCOL_SPEC.md 6.5).
inline constexpr uint8_t kDTypeInt8 = 1;
inline constexpr uint8_t kDTypeInt32 = 2;
inline constexpr uint8_t kMinRank = 1;
inline constexpr uint8_t kMaxRank = 4;

// Element size in bytes for a wire dtype (0 if the dtype code is unknown).
std::size_t dtypeSize(uint8_t dtype);

// Quant block (8 bytes at CRPK offset 16). Mirrors the .crml crml::quant*
// directives; workers use it for saturation bounds and host-side decode.
struct QuantBlock {
  uint8_t quantres = 0;
  uint8_t dtype = kDTypeInt32;  // 1 = int8, 2 = int32
  int16_t quantmin = 0;
  int16_t quantmax = 0;

  bool operator==(const QuantBlock &o) const {
    return quantres == o.quantres && dtype == o.dtype &&
           quantmin == o.quantmin && quantmax == o.quantmax;
  }
};

// One tensor table entry. `dims.size()` is the wire rank (1..4); `data` holds
// the row-major little-endian element bytes (dtypeSize(dtype) per element).
struct TensorEntry {
  uint8_t param_index = 0;
  uint8_t dtype = kDTypeInt32;
  std::vector<uint32_t> dims;
  std::vector<uint8_t> data;

  bool operator==(const TensorEntry &o) const {
    return param_index == o.param_index && dtype == o.dtype &&
           dims == o.dims && data == o.data;
  }
};

// A full execute request: quant block, opaque IR module bytes (embedded
// verbatim from caramel::ir::serialize()), and the input tensors in flow
// parameter declaration order (inputs[i].param_index == i).
struct CrpkRequest {
  QuantBlock quant;
  std::vector<uint8_t> ir;
  std::vector<TensorEntry> inputs;

  bool operator==(const CrpkRequest &o) const {
    return quant == o.quant && ir == o.ir && inputs == o.inputs;
  }
};

// A result envelope: status (0 = ok) and the output tensors in IR
// output-table order (outputs[i].param_index == i).
struct CrrsResponse {
  uint16_t status = 0;
  std::vector<TensorEntry> outputs;

  bool operator==(const CrrsResponse &o) const {
    return status == o.status && outputs == o.outputs;
  }
};

// Serialize a request to CRPK bytes. Throws std::invalid_argument /
// std::overflow_error on requests that cannot be encoded (bad dtype, rank
// outside 1..4, zero dims, data size not matching dims, sizes exceeding u32).
std::vector<uint8_t> serializeCrpk(const CrpkRequest &req);

// Parse CRPK bytes back into a request (nullopt on malformed input). Used for
// round-trip validation; the worker-side reference reader follows this logic.
std::optional<CrpkRequest> parseCrpk(const std::vector<uint8_t> &bytes);

// Serialize a response to CRRS bytes (same validation/throwing as
// serializeCrpk). crpk-gen uses this to emit the golden .crrs fixture.
std::vector<uint8_t> buildCrrs(const CrrsResponse &resp);

// Parse CRRS bytes into a response (nullopt on malformed input).
std::optional<CrrsResponse> parseCrrs(const std::vector<uint8_t> &bytes);

// --- Adapters: interpreter Value <-> wire TensorEntry ----------------------
// Encode an interpreter tensor as a wire entry (nullopt if the rank is not
// 1..4, a dim is non-positive or exceeds u32, or an element does not fit the
// wire dtype).
std::optional<TensorEntry> tensorFromValue(uint8_t param_index, uint8_t dtype,
                                           const caramel::interp::Value &v);

// Decode a wire entry into an interpreter tensor (nullopt on bad dtype or
// data size mismatch; elements are sign-extended into int64).
std::optional<caramel::interp::Value> valueFromTensor(const TensorEntry &t);

}  // namespace caramel::proto

#endif  // CARAMEL_PROTO_CRPK_H
