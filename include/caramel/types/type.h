// ============================================================================
// Caramel Language - Type system (tensors, scalars, quantization, shapes)
// ----------------------------------------------------------------------------
// Ticket:   lang_003 (Tensor Type System)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Caramel is an integer-only, fixed-point language: numeric literals are
// quantized to integers at compile time using the program's quant directives
// (the "decimal point removal" method). A value's runtime dtype is therefore an
// N-bit integer whose width is DERIVED from the quantization range, not declared.
// FP32 exists only at host encode/decode boundaries.
//
// This header defines:
//   * DType            - element data types (Int4/8/16/32, FP32-boundary)
//   * QuantParams      - quant range -> exact bit width, with bit-width calc
//   * Shape            - static or dynamic dimensions, with NumPy broadcasting
//   * Type hierarchy   - Scalar / Tensor / String / Text / Dictionary / Function
//   * Shape inference  - matmul and elementwise/broadcast inference helpers
//
// Per-operator type signatures live in lang_004; this header provides the
// representation and the inference primitives those signatures build on.
// See knowledge/compiler/TYPE_SYSTEM_DESIGN.md for rationale.
// ============================================================================
#ifndef CARAMEL_TYPES_TYPE_H
#define CARAMEL_TYPES_TYPE_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caramel::types {

// ---------------------------------------------------------------------------
// Element data types. The integer types are the only runtime compute types;
// FP32 marks host-boundary values (pre-encode / post-decode).
// ---------------------------------------------------------------------------
enum class DType { Int4, Int8, Int16, Int32, FP32 };

// Smallest standard integer DType that holds `bits` of signed range.
DType dtype_for_bits(int bits);

// Storage width in bits for a DType (Int4 reports 4; the allocator may still
// round up for CPU storage, but the FPGA uses the exact width).
int dtype_bits(DType dt);

const char* dtype_name(DType dt);

// ---------------------------------------------------------------------------
// Quantization parameters. Derived from the program's crml::quant* directives.
// The "decimal point removal" method:
//   multiplier   = 10^quantres
//   max_int      = round(quantmax * multiplier)
//   min_int      = round(quantmin * multiplier)
//   bit_width    = ceil(log2(max_int - min_int + 1))   (signed range)
// ---------------------------------------------------------------------------
struct QuantParams {
  int64_t max_int = 0;     // scaled integer maximum
  int64_t min_int = 0;     // scaled integer minimum
  int quantres = 0;        // decimal places preserved
  int64_t multiplier = 1;  // 10^quantres
  int bit_width = 0;       // exact signed bits needed

  // Build from human-facing range. quantmax/quantmin are real-valued bounds.
  static QuantParams from_range(double quantmax, double quantmin, int quantres);

  // The DType the compiler allocates for this signed integer range.
  DType dtype() const;
};

// ---------------------------------------------------------------------------
// Shapes. A dimension of kDynamic is unknown until runtime.
// ---------------------------------------------------------------------------
struct Shape {
  static constexpr int64_t kDynamic = -1;
  std::vector<int64_t> dims;

  Shape() = default;
  explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

  int rank() const { return static_cast<int>(dims.size()); }
  bool is_scalar() const { return dims.empty(); }
  bool is_static() const;                 // no kDynamic dims
  std::optional<int64_t> num_elements() const;  // only if fully static
  bool operator==(const Shape& other) const { return dims == other.dims; }
};

// NumPy-style broadcasting: align from the trailing dimension; dims must be
// equal, or one of them 1, or dynamic. Returns the broadcast shape or nullopt.
std::optional<Shape> broadcast(const Shape& a, const Shape& b);

// ---------------------------------------------------------------------------
// Type hierarchy. oc:: -> Scalar/Tensor, ot:: -> String/Text, os:: -> Dictionary.
// ---------------------------------------------------------------------------
enum class TypeKind { Scalar, Tensor, String, Text, Dictionary, Function, Unknown };

struct Type {
  explicit Type(TypeKind k) : kind(k) {}
  virtual ~Type() = default;
  TypeKind kind;
  virtual std::string str() const = 0;
};

struct ScalarType : Type {
  ScalarType() : Type(TypeKind::Scalar) {}
  DType dtype = DType::Int32;
  QuantParams quant{};
  std::string str() const override;
};

struct TensorType : Type {
  TensorType() : Type(TypeKind::Tensor) {}
  DType dtype = DType::Int32;
  Shape shape{};
  QuantParams quant{};
  std::string str() const override;  // e.g. "Tensor<Int8, [224, 224, 3]>"
};

struct StringType : Type {
  StringType() : Type(TypeKind::String) {}
  std::optional<int> trunc;  // chunk length, when truncation is specified
  std::string str() const override;
};

struct TextType : Type {
  TextType() : Type(TypeKind::Text) {}
  std::optional<int> trunc_char;
  std::optional<int> trunc_phrase;
  std::optional<int> trunc_paragraph;
  std::string str() const override;
};

struct DictionaryType : Type {
  DictionaryType() : Type(TypeKind::Dictionary) {}
  std::string str() const override;
};

struct UnknownType : Type {
  UnknownType() : Type(TypeKind::Unknown) {}
  std::string str() const override { return "<unknown>"; }
};

// ---------------------------------------------------------------------------
// Type-checking diagnostics
// ---------------------------------------------------------------------------
struct TypeError {
  std::string message;
};

template <typename T>
struct Inferred {
  std::optional<T> value;
  std::optional<TypeError> error;
  bool ok() const { return value.has_value(); }
  static Inferred good(T v) { return {std::move(v), std::nullopt}; }
  static Inferred bad(std::string msg) { return {std::nullopt, TypeError{std::move(msg)}}; }
};

// ---------------------------------------------------------------------------
// Shape inference helpers used by operator signatures (lang_004).
// ---------------------------------------------------------------------------
// matmul: [.., M, K] x [.., K, N] -> [.., M, N], with broadcasting on batch dims.
Inferred<Shape> infer_matmul(const Shape& a, const Shape& b);

// elementwise binary op: result is broadcast(a, b).
Inferred<Shape> infer_elementwise(const Shape& a, const Shape& b);

// Result quantization for an op that preserves resolution: the wider range of
// the inputs (used by add/sub/elementwise). lang_004 may override per op.
QuantParams combine_quant(const QuantParams& a, const QuantParams& b);

}  // namespace caramel::types

#endif  // CARAMEL_TYPES_TYPE_H
