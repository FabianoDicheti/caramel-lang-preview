// ============================================================================
// Caramel Language - Tensor constructors (generators) implementation
// ----------------------------------------------------------------------------
// See include/caramel/interp/generators.h.
// ============================================================================
#include "caramel/interp/generators.h"

#include <random>

namespace caramel::interp {

namespace {

struct GenSpec {
  const char *name;
  int arity;
};

// Arities are in RPN operand order. `diag`/`from_spectrum` take one vector.
const GenSpec kGenerators[] = {
    {"zeros", 2},    {"ones", 2},   {"full", 3},    {"eye", 1},
    {"diag", 1},     {"range", 1},  {"random", 2},  {"band", 4},
    {"tridiag", 4},  {"from_spectrum", 1},
};

bool as_scalar(const Value &v, int64_t &out, const std::string &op,
               int idx, std::string *err) {
  // A rank-1 tensor of one element is accepted as a scalar: reductions in this
  // language return [1], so `x tensor_max n eye` would otherwise be a rank error.
  if (v.is_scalar() || (v.rank() == 1 && v.data.size() == 1)) {
    out = v.data.empty() ? 0 : v.data[0];
    return true;
  }
  if (err)
    *err = op + ": argument " + std::to_string(idx + 1) +
           " must be a scalar, got rank " + std::to_string(v.rank());
  return false;
}

bool as_vector(const Value &v, std::vector<int64_t> &out, const std::string &op,
               std::string *err) {
  if (v.rank() != 1) {
    if (err)
      *err = op + ": argument must be a rank-1 vector, got rank " +
             std::to_string(v.rank());
    return false;
  }
  out = v.data;
  return true;
}

bool check_dim(int64_t n, const std::string &op, std::string *err) {
  if (n < 0) {
    if (err) *err = op + ": negative dimension (" + std::to_string(n) + ")";
    return false;
  }
  // Guard the allocation: a typo'd dimension should be an error, not an OOM.
  if (n > 100000) {
    if (err)
      *err = op + ": dimension " + std::to_string(n) + " is unreasonably large";
    return false;
  }
  return true;
}

}  // namespace

int generatorArity(const std::string &op) {
  for (const auto &g : kGenerators)
    if (op == g.name) return g.arity;
  return -1;
}

bool isGenerator(const std::string &op) { return generatorArity(op) >= 0; }

std::optional<Value> evalGenerator(const std::string &op,
                                   const std::vector<Value> &args,
                                   std::string *err) {
  const int arity = generatorArity(op);
  if (arity < 0) return std::nullopt;   // not a constructor: caller tries elsewhere
  if ((int)args.size() != arity) {
    if (err)
      *err = op + " expects " + std::to_string(arity) + " argument(s), got " +
             std::to_string(args.size());
    return std::nullopt;
  }

  if (op == "zeros" || op == "ones") {
    int64_t r, c;
    if (!as_scalar(args[0], r, op, 0, err) || !as_scalar(args[1], c, op, 1, err))
      return std::nullopt;
    if (!check_dim(r, op, err) || !check_dim(c, op, err)) return std::nullopt;
    return Value::tensor({r, c}, std::vector<int64_t>((size_t)(r * c),
                                                      op == "ones" ? 1 : 0));
  }
  if (op == "full") {
    int64_t r, c, v;
    if (!as_scalar(args[0], r, op, 0, err) || !as_scalar(args[1], c, op, 1, err) ||
        !as_scalar(args[2], v, op, 2, err))
      return std::nullopt;
    if (!check_dim(r, op, err) || !check_dim(c, op, err)) return std::nullopt;
    return Value::tensor({r, c}, std::vector<int64_t>((size_t)(r * c), v));
  }
  if (op == "eye") {
    int64_t n;
    if (!as_scalar(args[0], n, op, 0, err)) return std::nullopt;
    if (!check_dim(n, op, err)) return std::nullopt;
    std::vector<int64_t> d((size_t)(n * n), 0);
    for (int64_t i = 0; i < n; ++i) d[(size_t)(i * n + i)] = 1;
    return Value::tensor({n, n}, std::move(d));
  }
  if (op == "diag" || op == "from_spectrum") {
    // Canonical diagonal witness: a square matrix carrying the given vector on
    // its main diagonal. For from_spectrum this is the well-posed construction —
    // a diagonal matrix's eigenvalues ARE its diagonal, so the result provably
    // has the requested spectrum.
    std::vector<int64_t> vec;
    if (!as_vector(args[0], vec, op, err)) return std::nullopt;
    const int64_t n = (int64_t)vec.size();
    if (!check_dim(n, op, err)) return std::nullopt;
    std::vector<int64_t> d((size_t)(n * n), 0);
    for (int64_t i = 0; i < n; ++i) d[(size_t)(i * n + i)] = vec[(size_t)i];
    return Value::tensor({n, n}, std::move(d));
  }
  if (op == "range") {
    int64_t n;
    if (!as_scalar(args[0], n, op, 0, err)) return std::nullopt;
    if (!check_dim(n, op, err)) return std::nullopt;
    std::vector<int64_t> d((size_t)n);
    for (int64_t i = 0; i < n; ++i) d[(size_t)i] = i;
    return Value::tensor({n}, std::move(d));
  }
  if (op == "random") {
    int64_t r, c;
    if (!as_scalar(args[0], r, op, 0, err) || !as_scalar(args[1], c, op, 1, err))
      return std::nullopt;
    if (!check_dim(r, op, err) || !check_dim(c, op, err)) return std::nullopt;
    // Deterministic (fixed seed) so runs are reproducible; integer range
    // [0, 100). Seed/range parameterization is a later extension.
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, 99);
    std::vector<int64_t> d((size_t)(r * c));
    for (auto &e : d) e = dist(rng);
    return Value::tensor({r, c}, std::move(d));
  }
  if (op == "band") {
    int64_t n, lower, upper, fill;
    if (!as_scalar(args[0], n, op, 0, err) ||
        !as_scalar(args[1], lower, op, 1, err) ||
        !as_scalar(args[2], upper, op, 2, err) ||
        !as_scalar(args[3], fill, op, 3, err))
      return std::nullopt;
    if (!check_dim(n, op, err)) return std::nullopt;
    std::vector<int64_t> d((size_t)(n * n), 0);
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < n; ++j)
        if ((i - j) <= lower && (j - i) <= upper)
          d[(size_t)(i * n + j)] = fill;
    return Value::tensor({n, n}, std::move(d));
  }
  if (op == "tridiag") {
    int64_t n, sub, diagv, super;
    if (!as_scalar(args[0], n, op, 0, err) ||
        !as_scalar(args[1], sub, op, 1, err) ||
        !as_scalar(args[2], diagv, op, 2, err) ||
        !as_scalar(args[3], super, op, 3, err))
      return std::nullopt;
    if (!check_dim(n, op, err)) return std::nullopt;
    std::vector<int64_t> d((size_t)(n * n), 0);
    for (int64_t i = 0; i < n; ++i) {
      d[(size_t)(i * n + i)] = diagv;
      if (i > 0) d[(size_t)(i * n + (i - 1))] = sub;
      if (i + 1 < n) d[(size_t)(i * n + (i + 1))] = super;
    }
    return Value::tensor({n, n}, std::move(d));
  }

  if (err) *err = "unhandled tensor constructor '" + op + "'";
  return std::nullopt;
}

}  // namespace caramel::interp
