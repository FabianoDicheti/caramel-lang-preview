// ============================================================================
// Caramel Language - Type system implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_003 (Tensor Type System)
// Version: 1.0.0
// ============================================================================
#include "caramel/types/type.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace caramel::types {

namespace {

int signed_bits_for_range(int64_t min_int, int64_t max_int) {
  int bits = 1;
  while (bits < 63) {
    const int64_t low = -(int64_t{1} << (bits - 1));
    const int64_t high = (int64_t{1} << (bits - 1)) - 1;
    if (min_int >= low && max_int <= high) return bits;
    ++bits;
  }
  return 63;
}

}  // namespace

int dtype_bits(DType dt) {
  switch (dt) {
    case DType::Int4:  return 4;
    case DType::Int8:  return 8;
    case DType::Int16: return 16;
    case DType::Int32: return 32;
    case DType::FP32:  return 32;
  }
  return 32;
}

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::Int4:  return "Int4";
    case DType::Int8:  return "Int8";
    case DType::Int16: return "Int16";
    case DType::Int32: return "Int32";
    case DType::FP32:  return "FP32";
  }
  return "Int32";
}

DType dtype_for_bits(int bits) {
  if (bits <= 4) return DType::Int4;
  if (bits <= 8) return DType::Int8;
  if (bits <= 16) return DType::Int16;
  return DType::Int32;
}

QuantParams QuantParams::from_range(double quantmax, double quantmin, int quantres) {
  QuantParams q;
  q.quantres = quantres;
  q.multiplier = 1;
  for (int i = 0; i < quantres; ++i) q.multiplier *= 10;
  q.max_int = static_cast<int64_t>(std::llround(quantmax * static_cast<double>(q.multiplier)));
  q.min_int = static_cast<int64_t>(std::llround(quantmin * static_cast<double>(q.multiplier)));
  // Pick enough bits for the actual signed endpoints, not just the number of
  // representable values in the span. Asymmetric ranges such as [-200, 55] need
  // 9 signed bits even though their cardinality is 256.
  const int64_t span = q.max_int - q.min_int;
  const int64_t cardinality = (span >= 0 ? span : -span) + 1;
  int bits = 1;
  int64_t capacity = 2;  // 2^bits
  while (capacity < cardinality) {
    capacity <<= 1;
    ++bits;
  }
  q.bit_width = std::max(bits, signed_bits_for_range(q.min_int, q.max_int));
  return q;
}

DType QuantParams::dtype() const { return dtype_for_bits(bit_width); }

bool Shape::is_static() const {
  return std::none_of(dims.begin(), dims.end(),
                      [](int64_t d) { return d == kDynamic; });
}

std::optional<int64_t> Shape::num_elements() const {
  if (!is_static()) return std::nullopt;
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  return n;
}

std::optional<Shape> broadcast(const Shape& a, const Shape& b) {
  const int ra = a.rank(), rb = b.rank();
  const int r = std::max(ra, rb);
  std::vector<int64_t> out(r);
  for (int i = 0; i < r; ++i) {
    const int64_t da = (i < r - ra) ? 1 : a.dims[i - (r - ra)];
    const int64_t db = (i < r - rb) ? 1 : b.dims[i - (r - rb)];
    if (da == Shape::kDynamic || db == Shape::kDynamic) {
      out[i] = Shape::kDynamic;
    } else if (da == db) {
      out[i] = da;
    } else if (da == 1) {
      out[i] = db;
    } else if (db == 1) {
      out[i] = da;
    } else {
      return std::nullopt;  // incompatible
    }
  }
  return Shape(std::move(out));
}

std::string ScalarType::str() const {
  std::ostringstream os;
  os << "Scalar<" << dtype_name(dtype) << ">";
  return os.str();
}

std::string TensorType::str() const {
  std::ostringstream os;
  os << "Tensor<" << dtype_name(dtype) << ", [";
  for (size_t i = 0; i < shape.dims.size(); ++i) {
    if (i) os << ", ";
    if (shape.dims[i] == Shape::kDynamic) os << "?";
    else os << shape.dims[i];
  }
  os << "]>";
  return os.str();
}

std::string StringType::str() const {
  return trunc ? "String<trunc=" + std::to_string(*trunc) + ">" : "String";
}

std::string TextType::str() const { return "Text"; }

std::string DictionaryType::str() const { return "Dictionary"; }

Inferred<Shape> infer_matmul(const Shape& a, const Shape& b) {
  if (a.rank() < 2 || b.rank() < 2) {
    return Inferred<Shape>::bad("matmul requires rank >= 2 operands");
  }
  const int64_t ak = a.dims[a.rank() - 1];
  const int64_t bk = b.dims[b.rank() - 2];
  if (ak != Shape::kDynamic && bk != Shape::kDynamic && ak != bk) {
    return Inferred<Shape>::bad(
        "matmul inner dimensions disagree: " + std::to_string(ak) + " vs " +
        std::to_string(bk));
  }
  const int64_t m = a.dims[a.rank() - 2];
  const int64_t n = b.dims[b.rank() - 1];

  // Broadcast leading batch dimensions.
  Shape batch_a(std::vector<int64_t>(a.dims.begin(), a.dims.end() - 2));
  Shape batch_b(std::vector<int64_t>(b.dims.begin(), b.dims.end() - 2));
  auto batch = broadcast(batch_a, batch_b);
  if (!batch) return Inferred<Shape>::bad("matmul batch dimensions not broadcastable");

  std::vector<int64_t> out = batch->dims;
  out.push_back(m);
  out.push_back(n);
  return Inferred<Shape>::good(Shape(std::move(out)));
}

Inferred<Shape> infer_elementwise(const Shape& a, const Shape& b) {
  if (auto s = broadcast(a, b)) return Inferred<Shape>::good(*s);
  return Inferred<Shape>::bad("elementwise operands not broadcastable");
}

QuantParams combine_quant(const QuantParams& a, const QuantParams& b) {
  // Preserve the finer resolution and the wider integer range.
  QuantParams q;
  q.quantres = std::max(a.quantres, b.quantres);
  q.multiplier = std::max(a.multiplier, b.multiplier);
  q.max_int = std::max(a.max_int, b.max_int);
  q.min_int = std::min(a.min_int, b.min_int);
  const int64_t span = q.max_int - q.min_int;
  const int64_t cardinality = (span >= 0 ? span : -span) + 1;
  int bits = 1;
  int64_t capacity = 2;
  while (capacity < cardinality) {
    capacity <<= 1;
    ++bits;
  }
  q.bit_width = std::max(bits, signed_bits_for_range(q.min_int, q.max_int));
  return q;
}

}  // namespace caramel::types
