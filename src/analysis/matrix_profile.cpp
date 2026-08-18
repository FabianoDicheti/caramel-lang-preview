// ============================================================================
// Caramel Language - Matrix algebraic profiler implementation
// ----------------------------------------------------------------------------
// See include/caramel/analysis/matrix_profile.h. Tier 1 is exact integer work;
// Tier 2 uses double internally (off the runtime hot path). Numeric strategy:
//   * determinant: fraction-free Bareiss over __int128 (exact when it fits),
//     double LU fallback otherwise;
//   * rank: double Gaussian elimination with partial pivoting;
//   * eigenvalues: exact diagonal read for triangular/diagonal, cyclic Jacobi
//     for symmetric, unshifted QR iteration (best-effort) for general matrices.
// ============================================================================
#include "caramel/analysis/matrix_profile.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace caramel::analysis {

namespace {

using Mat = std::vector<std::vector<double>>;

// __int128 decimal string (handles the exact determinant).
std::string i128_to_string(__int128 v) {
  if (v == 0) return "0";
  bool neg = v < 0;
  unsigned __int128 u = neg ? (unsigned __int128)(-(v + 1)) + 1u : (unsigned __int128)v;
  std::string s;
  while (u > 0) { s.push_back(char('0' + int(u % 10))); u /= 10; }
  if (neg) s.push_back('-');
  std::reverse(s.begin(), s.end());
  return s;
}

// Fraction-free Bareiss determinant over 128-bit integers. Returns false when a
// factor grows past a safe magnitude (caller falls back to a double estimate).
bool bareiss_det(std::vector<std::vector<__int128>> a, int n, __int128 &out) {
  const __int128 kLimit = (__int128)1000000000000000000LL;  // 1e18 guard
  __int128 prev = 1;
  int sign = 1;
  for (int k = 0; k < n - 1; ++k) {
    if (a[k][k] == 0) {
      int p = -1;
      for (int i = k + 1; i < n; ++i) if (a[i][k] != 0) { p = i; break; }
      if (p == -1) { out = 0; return true; }  // zero column -> singular
      std::swap(a[k], a[p]);
      sign = -sign;
    }
    for (int i = k + 1; i < n; ++i)
      for (int j = k + 1; j < n; ++j) {
        __int128 aij = a[i][j], akk = a[k][k], aik = a[i][k], akj = a[k][j];
        if (aij > kLimit || aij < -kLimit || akk > kLimit || akk < -kLimit ||
            aik > kLimit || aik < -kLimit || akj > kLimit || akj < -kLimit)
          return false;
        a[i][j] = (aij * akk - aik * akj) / prev;  // exact by Bareiss
      }
    prev = a[k][k];
  }
  out = (__int128)sign * a[n - 1][n - 1];
  return true;
}

// Magnitude guard shared by the exact integer kernels: 1e18 leaves room for one
// product (1e36) inside __int128's ~1.7e38 range.
inline bool i128_too_big(__int128 v) {
  const __int128 kLimit = (__int128)1000000000000000000LL;
  return v > kLimit || v < -kLimit;
}

// Exact rank by fraction-free (Bareiss) elimination over 128-bit integers.
// Every division is exact in theory; we assert it with a remainder check and
// bail out rather than trust the theory, so a wrong rank can never be reported
// as exact. Returns false on overflow or a non-exact division (caller falls
// back to the double path), true with `out` set otherwise.
bool exact_rank(std::vector<std::vector<__int128>> a, int r, int c, int &out) {
  __int128 prev = 1;
  int row = 0;
  for (int col = 0; col < c && row < r; ++col) {
    int piv = -1;
    for (int i = row; i < r; ++i)
      if (a[i][col] != 0) { piv = i; break; }
    if (piv < 0) continue;  // whole column below `row` is zero: not a pivot
    if (piv != row) std::swap(a[row], a[piv]);
    for (int i = row + 1; i < r; ++i) {
      for (int j = col + 1; j < c; ++j) {
        __int128 aij = a[i][j], ap = a[row][col], aic = a[i][col], apj = a[row][j];
        if (i128_too_big(aij) || i128_too_big(ap) || i128_too_big(aic) ||
            i128_too_big(apj))
          return false;
        __int128 num = aij * ap - aic * apj;
        if (num % prev != 0) return false;  // division was not exact: distrust
        a[i][j] = num / prev;
      }
      a[i][col] = 0;
    }
    prev = a[row][col];
    ++row;
  }
  out = row;
  return true;
}

// Copy an integer matrix into __int128 working storage.
std::vector<std::vector<__int128>> to_i128(const std::vector<int64_t> &X, int r,
                                           int c) {
  std::vector<std::vector<__int128>> a(r, std::vector<__int128>(c));
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) a[i][j] = X[(size_t)i * c + j];
  return a;
}

// Exact n x n integer product with an overflow guard.
bool i128_matmul(const std::vector<std::vector<__int128>> &A,
                 const std::vector<std::vector<__int128>> &B, int n,
                 std::vector<std::vector<__int128>> &out) {
  out.assign(n, std::vector<__int128>(n, 0));
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      __int128 a = A[i][k];
      if (a == 0) continue;
      if (i128_too_big(a)) return false;
      for (int j = 0; j < n; ++j) {
        if (i128_too_big(B[k][j]) || i128_too_big(out[i][j])) return false;
        out[i][j] += a * B[k][j];
      }
    }
  return true;
}

double lu_det(Mat a, int n) {
  double det = 1.0;
  for (int k = 0; k < n; ++k) {
    int piv = k;
    for (int i = k + 1; i < n; ++i)
      if (std::fabs(a[i][k]) > std::fabs(a[piv][k])) piv = i;
    if (std::fabs(a[piv][k]) < 1e-300) return 0.0;
    if (piv != k) { std::swap(a[k], a[piv]); det = -det; }
    det *= a[k][k];
    for (int i = k + 1; i < n; ++i) {
      double f = a[i][k] / a[k][k];
      for (int j = k; j < n; ++j) a[i][j] -= f * a[k][j];
    }
  }
  return det;
}

// Rank via Gaussian elimination with partial pivoting (double).
int rank_of(Mat a, int r, int c, double tol) {
  int rank = 0, row = 0;
  for (int col = 0; col < c && row < r; ++col) {
    int piv = -1;
    double best = tol;
    for (int i = row; i < r; ++i)
      if (std::fabs(a[i][col]) > best) { best = std::fabs(a[i][col]); piv = i; }
    if (piv < 0) continue;
    std::swap(a[row], a[piv]);
    for (int i = 0; i < r; ++i) {
      if (i == row) continue;
      double f = a[i][col] / a[row][col];
      for (int j = col; j < c; ++j) a[i][j] -= f * a[row][j];
    }
    ++row; ++rank;
  }
  return rank;
}

// Cyclic Jacobi eigenvalues for a symmetric matrix (double).
std::vector<double> jacobi_eigenvalues(Mat a, int n, double tol) {
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += a[p][q] * a[p][q];
    if (off < tol * tol) break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        if (std::fabs(a[p][q]) < 1e-300) continue;
        double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        double t = (theta >= 0 ? 1.0 : -1.0) /
                   (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        double cth = 1.0 / std::sqrt(t * t + 1.0), sth = t * cth;
        for (int i = 0; i < n; ++i) {
          double aip = a[i][p], aiq = a[i][q];
          a[i][p] = cth * aip - sth * aiq;
          a[i][q] = sth * aip + cth * aiq;
        }
        for (int i = 0; i < n; ++i) {
          double api = a[p][i], aqi = a[q][i];
          a[p][i] = cth * api - sth * aqi;
          a[q][i] = sth * api + cth * aqi;
        }
      }
  }
  std::vector<double> ev(n);
  for (int i = 0; i < n; ++i) ev[i] = a[i][i];
  return ev;
}

// Unshifted QR iteration (Gram-Schmidt) for a general matrix. Best-effort:
// converges to real eigenvalues on the diagonal for real, separable spectra.
// `converged` is set false if the subdiagonal mass does not decay (a hint that
// the spectrum may be complex / unreliable).
std::vector<double> qr_eigenvalues(Mat a, int n, double tol, bool &converged) {
  converged = true;
  for (int iter = 0; iter < 500; ++iter) {
    // Gram-Schmidt QR of a: a = Q R.
    Mat Q(n, std::vector<double>(n, 0.0)), R(n, std::vector<double>(n, 0.0));
    for (int j = 0; j < n; ++j) {
      std::vector<double> v(n);
      for (int i = 0; i < n; ++i) v[i] = a[i][j];
      for (int k = 0; k < j; ++k) {
        double dot = 0.0;
        for (int i = 0; i < n; ++i) dot += Q[i][k] * a[i][j];
        R[k][j] = dot;
        for (int i = 0; i < n; ++i) v[i] -= dot * Q[i][k];
      }
      double nrm = 0.0;
      for (int i = 0; i < n; ++i) nrm += v[i] * v[i];
      nrm = std::sqrt(nrm);
      R[j][j] = nrm;
      if (nrm < 1e-300) { for (int i = 0; i < n; ++i) Q[i][j] = 0.0; continue; }
      for (int i = 0; i < n; ++i) Q[i][j] = v[i] / nrm;
    }
    // a = R Q.
    Mat na(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int k = 0; k < n; ++k) s += R[i][k] * Q[k][j];
        na[i][j] = s;
      }
    a = std::move(na);
    double sub = 0.0;
    for (int i = 1; i < n; ++i) sub += std::fabs(a[i][i - 1]);
    if (sub < tol) { converged = true; break; }
    if (iter == 499) converged = (sub < 1e-3);
  }
  std::vector<double> ev(n);
  for (int i = 0; i < n; ++i) ev[i] = a[i][i];
  return ev;
}

// Exact integer A*B (n x n), returned as __int128 to avoid overflow on compare.
std::vector<std::vector<__int128>> imatmul(const std::vector<int64_t> &A,
                                           const std::vector<int64_t> &B, int n) {
  std::vector<std::vector<__int128>> C(n, std::vector<__int128>(n, 0));
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      __int128 a = A[i * n + k];
      if (a == 0) continue;
      for (int j = 0; j < n; ++j) C[i][j] += a * (__int128)B[k * n + j];
    }
  return C;
}

// Dense double matmul, used only for the (A - lambda I)^k rank sequence.
Mat dmatmul(const Mat &A, const Mat &B, int n) {
  Mat C(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      const double a = A[i][k];
      if (a == 0.0) continue;
      for (int j = 0; j < n; ++j) C[i][j] += a * B[k][j];
    }
  return C;
}

// Jordan block sizes for one eigenvalue, from the rank sequence
//   r_k = rank((A - lambda I)^k),  r_0 = n.
// The number of blocks of size exactly k is r_{k-1} - 2*r_k + r_{k+1}, and the
// sequence stabilises once r_k stops decreasing (k > algebraic multiplicity).
std::vector<int> jordan_sizes(const Mat &A, int n, double lambda, int alg,
                              double tol) {
  Mat N(n, std::vector<double>(n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) N[i][j] = A[i][j] - (i == j ? lambda : 0.0);

  std::vector<int> r;         // r[k] = rank(N^k), starting at k = 0
  r.push_back(n);
  Mat P = N;
  for (int k = 1; k <= alg + 1; ++k) {
    r.push_back(rank_of(P, n, n, tol));
    if (r[k] == r[k - 1]) break;   // stabilised: further powers add nothing
    if (k <= alg) P = dmatmul(P, N, n);
  }
  // pad so r_{k-1}, r_k, r_{k+1} are all readable
  while ((int)r.size() < alg + 2) r.push_back(r.back());

  std::vector<int> sizes;
  for (int k = 1; k <= alg; ++k) {
    int count = r[k - 1] - 2 * r[k] + r[k + 1];
    for (int c = 0; c < count; ++c) sizes.push_back(k);
  }
  return sizes;
}

// Everything exactly knowable about an INTEGER candidate eigenvalue.
struct ExactEigen {
  int algebraic = 0;
  int geometric = 0;
  std::vector<int> jordan;   // block sizes, unsorted
};

// Certify an integer lambda against an integer matrix and derive its exact
// multiplicities, entirely in integer arithmetic. (A - lambda I)^k never leaves
// the integers, so the whole rank sequence
//     r_0 = n,  r_k = rank((A - lambda I)^k)
// is exact, and from it:
//   * lambda is an eigenvalue  <=>  r_1 < n   (equivalently det(A - lambda I) = 0)
//   * geometric multiplicity   =   n - r_1          (the eigenspace)
//   * algebraic multiplicity   =   n - r_stable     (the generalized eigenspace,
//     which is what the characteristic polynomial counts)
//   * blocks of size exactly k =   r_{k-1} - 2 r_k + r_{k+1}
//
// Taking the algebraic multiplicity from the rank sequence rather than from
// float grouping of QR output is the point: it makes the multiplicity a proof
// rather than a tolerance judgement.
//
// Returns false if lambda is not an eigenvalue, or if the exact kernels hit
// their overflow guard (caller must then fall back and stay tagged inexact).
bool certify_integer_eigenvalue(const std::vector<int64_t> &X, int n,
                                int64_t lambda, ExactEigen &out) {
  auto N = to_i128(X, n, n);
  for (int i = 0; i < n; ++i) N[i][i] -= lambda;

  std::vector<int> r;
  r.push_back(n);
  auto P = N;
  for (int k = 1; k <= n + 1; ++k) {
    int rk = 0;
    if (!exact_rank(P, n, n, rk)) return false;
    r.push_back(rk);
    if (r[k] == r[k - 1]) break;   // stabilised: further powers add nothing
    std::vector<std::vector<__int128>> next;
    if (!i128_matmul(P, N, n, next)) return false;
    P = std::move(next);
  }
  if (r.size() < 2 || r[1] == n) return false;  // not an eigenvalue

  const int stable = r.back();
  out.geometric = n - r[1];
  out.algebraic = n - stable;
  if (out.algebraic < out.geometric) return false;  // impossible: distrust

  // pad so r_{k-1}, r_k, r_{k+1} are readable for every k we report on
  while ((int)r.size() < out.algebraic + 2) r.push_back(r.back());
  out.jordan.clear();
  for (int k = 1; k <= out.algebraic; ++k) {
    int count = r[k - 1] - 2 * r[k] + r[k + 1];
    for (int c = 0; c < count; ++c) out.jordan.push_back(k);
  }
  return true;
}

void push_unique(std::vector<std::string> &v, const std::string &s) {
  if (std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
}

}  // namespace

// ---------------------------------------------------------------------------
// Taxonomy: predicate vocabulary + rule base
// ---------------------------------------------------------------------------
namespace {

struct PredicateEntry {
  const char *name;
  bool (*test)(const MatrixDescriptor &);
};

// The vocabulary a rule may name. Compound predicates ("strictly_lower_
// triangular" = lower triangular but not diagonal) live here rather than as
// conjunctions inside a rule, so that every rule keeps a single antecedent and
// stays mechanically invertible.
const PredicateEntry kPredicates[] = {
    {"square",        [](const MatrixDescriptor &d) { return d.is_square; }},
    {"identity",      [](const MatrixDescriptor &d) { return d.is_identity; }},
    {"zero",          [](const MatrixDescriptor &d) { return d.is_zero; }},
    {"diagonal",      [](const MatrixDescriptor &d) { return d.is_diagonal; }},
    {"lower_triangular", [](const MatrixDescriptor &d) { return d.is_lower_triangular; }},
    {"upper_triangular", [](const MatrixDescriptor &d) { return d.is_upper_triangular; }},
    {"strictly_lower_triangular",
     [](const MatrixDescriptor &d) { return d.is_lower_triangular && !d.is_diagonal; }},
    {"strictly_upper_triangular",
     [](const MatrixDescriptor &d) { return d.is_upper_triangular && !d.is_diagonal; }},
    {"triangular_or_diagonal",
     [](const MatrixDescriptor &d) {
       return d.is_lower_triangular || d.is_upper_triangular || d.is_diagonal;
     }},
    {"band",          [](const MatrixDescriptor &d) { return d.is_band; }},
    {"bidiagonal",    [](const MatrixDescriptor &d) { return d.is_bidiagonal; }},
    {"tridiagonal",   [](const MatrixDescriptor &d) { return d.is_tridiagonal; }},
    {"sparse",        [](const MatrixDescriptor &d) { return d.is_sparse; }},
    {"dense",         [](const MatrixDescriptor &d) { return d.is_dense; }},
    {"symmetric",     [](const MatrixDescriptor &d) { return d.is_symmetric; }},
    {"skew_symmetric",[](const MatrixDescriptor &d) { return d.is_skew_symmetric; }},
    {"orthogonal",    [](const MatrixDescriptor &d) { return d.is_orthogonal; }},
    {"normal",        [](const MatrixDescriptor &d) { return d.is_normal; }},
    {"full_rank",     [](const MatrixDescriptor &d) { return d.has_rank && d.is_full_rank; }},
    {"invertible",    [](const MatrixDescriptor &d) { return d.is_invertible; }},
    {"singular",      [](const MatrixDescriptor &d) { return d.has_rank && d.is_singular; }},
    {"defective",     [](const MatrixDescriptor &d) { return d.is_defective; }},
    {"diagonalizable",
     [](const MatrixDescriptor &d) { return d.spectrum_determined && d.is_diagonalizable; }},
    {"positive_definite", [](const MatrixDescriptor &d) { return d.is_positive_definite; }},
    {"spd",
     [](const MatrixDescriptor &d) { return d.is_symmetric && d.is_positive_definite; }},
    {"nilpotent",     [](const MatrixDescriptor &d) { return d.is_nilpotent; }},
    {"idempotent",    [](const MatrixDescriptor &d) { return d.is_idempotent; }},
    {"involutory",    [](const MatrixDescriptor &d) { return d.is_involutory; }},
    {"spectrum_determined",
     [](const MatrixDescriptor &d) { return d.spectrum_determined; }},
};

// The rule base. Read forward by the profiler; meant to be read backward by the
// synthesis engine to reject self-contradictory property specs.
const std::vector<TaxonomyRule> kRules = {
    // --- inherits: the derived-class tags a matrix carries -----------------
    {"square",                    RuleKind::Inherits, "square_matrix"},
    {"!square",                   RuleKind::Inherits, "rectangular_matrix"},
    {"identity",                  RuleKind::Inherits, "identity_matrix"},
    {"zero",                      RuleKind::Inherits, "zero_matrix"},
    {"diagonal",                  RuleKind::Inherits, "diagonal_matrix"},
    {"strictly_lower_triangular", RuleKind::Inherits, "lower_triangular_matrix"},
    {"strictly_upper_triangular", RuleKind::Inherits, "upper_triangular_matrix"},
    {"band",                      RuleKind::Inherits, "band_matrix"},
    {"bidiagonal",                RuleKind::Inherits, "bidiagonal_matrix"},
    {"tridiagonal",               RuleKind::Inherits, "tridiagonal_matrix"},
    {"sparse",                    RuleKind::Inherits, "sparse_matrix"},
    {"dense",                     RuleKind::Inherits, "dense_matrix"},
    {"symmetric",                 RuleKind::Inherits, "symmetric_matrix"},
    {"skew_symmetric",            RuleKind::Inherits, "skew_symmetric_matrix"},
    {"orthogonal",                RuleKind::Inherits, "orthogonal_matrix"},
    {"full_rank",                 RuleKind::Inherits, "full_rank_matrix"},
    {"invertible",                RuleKind::Inherits, "invertible_matrix"},
    {"singular",                  RuleKind::Inherits, "singular_matrix"},
    {"defective",                 RuleKind::Inherits, "defective_matrix"},
    {"diagonalizable",            RuleKind::Inherits, "diagonalizable_matrix"},
    {"positive_definite",         RuleKind::Inherits, "positive_definite_matrix"},
    {"nilpotent",                 RuleKind::Inherits, "nilpotent_matrix"},
    {"idempotent",                RuleKind::Inherits, "idempotent_matrix"},
    {"involutory",                RuleKind::Inherits, "involutory_matrix"},

    // --- implies: forced facts (the profiler cross-checks these) -----------
    {"triangular_or_diagonal",    RuleKind::Implies, "eigenvalues_equal_diagonal"},
    {"triangular_or_diagonal",    RuleKind::Implies, "determinant_equals_diagonal_product"},
    {"invertible",                RuleKind::Implies, "full_rank"},
    {"full_rank",                 RuleKind::Implies, "invertible"},
    {"symmetric",                 RuleKind::Implies, "real_spectrum"},
    {"symmetric",                 RuleKind::Implies, "normal"},
    {"orthogonal",                RuleKind::Implies, "determinant_is_plus_or_minus_one"},
    {"orthogonal",                RuleKind::Implies, "normal"},
    {"diagonal",                  RuleKind::Implies, "symmetric"},
    {"spd",                       RuleKind::Implies, "invertible"},
    {"defective",                 RuleKind::Implies, "not_diagonalizable"},
    {"identity",                  RuleKind::Implies, "involutory"},

    // --- forbids: contradiction detection (the phase-2 lever) -------------
    {"!symmetric",                RuleKind::Forbids, "symmetric"},
    {"!diagonal",                 RuleKind::Forbids, "diagonal"},
    {"!orthogonal",               RuleKind::Forbids, "orthogonal"},
    {"!spd",                      RuleKind::Forbids, "spd"},
    {"invertible",                RuleKind::Forbids, "singular"},
    {"singular",                  RuleKind::Forbids, "invertible"},
    {"defective",                 RuleKind::Forbids, "diagonalizable"},
    {"diagonalizable",            RuleKind::Forbids, "defective"},
    // General lattice rules, not necessarily triggered by any one fixture:
    // an orthogonal matrix is normal, hence diagonalizable, hence not defective.
    {"orthogonal",                RuleKind::Forbids, "defective"},
    {"symmetric",                 RuleKind::Forbids, "defective"},
    {"nilpotent",                 RuleKind::Forbids, "invertible"},
    {"identity",                  RuleKind::Forbids, "singular"},
};

}  // namespace

const std::vector<TaxonomyRule> &taxonomyRules() { return kRules; }

bool evalTaxonomyPredicate(const std::string &name, const MatrixDescriptor &d,
                           bool &out) {
  bool negate = false;
  std::string key = name;
  if (!key.empty() && key[0] == '!') { negate = true; key.erase(0, 1); }
  for (const auto &p : kPredicates) {
    if (key == p.name) {
      out = negate ? !p.test(d) : p.test(d);
      return true;
    }
  }
  return false;
}

const std::vector<std::string> &taxonomyPredicateNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v;
    for (const auto &p : kPredicates) v.push_back(p.name);
    return v;
  }();
  return names;
}

MatrixDescriptor profile(const interp::Value &m, const std::string &name,
                         const ProfileOptions &opts) {
  MatrixDescriptor d;
  d.name = name;
  if (m.rank() != 2) {
    d.error = "profile expects a rank-2 matrix (got rank " +
              std::to_string(m.rank()) + ")";
    return d;
  }
  const int64_t R = m.dims[0], C = m.dims[1];
  const auto &X = m.data;
  auto at = [&](int64_t i, int64_t j) -> int64_t { return X[i * C + j]; };
  d.ok = true;
  d.rows = R; d.cols = C;
  d.is_square = (R == C);
  d.is_rectangular = (R != C);
  d.is_row_matrix = (R == 1);
  d.is_column_matrix = (C == 1);

  // --- Tier 1: sparsity + bandwidths + structure -------------------------
  int64_t nnz = 0, bw_up = 0, bw_lo = 0;
  for (int64_t i = 0; i < R; ++i)
    for (int64_t j = 0; j < C; ++j)
      if (at(i, j) != 0) {
        ++nnz;
        if (j > i) bw_up = std::max(bw_up, j - i);
        if (i > j) bw_lo = std::max(bw_lo, i - j);
      }
  d.non_zero = nnz;
  d.zero_count = R * C - nnz;
  d.sparsity_num = d.zero_count;
  d.sparsity_den = R * C;
  d.sparsity_ratio = R * C ? double(d.zero_count) / double(R * C) : 0.0;
  d.is_sparse = d.sparsity_ratio >= 0.5;
  d.is_dense = !d.is_sparse;
  d.bandwidth_upper = bw_up;
  d.bandwidth_lower = bw_lo;
  d.total_bandwidth = bw_up + bw_lo;

  if (d.is_square) {
    d.is_upper_triangular = (bw_lo == 0);
    d.is_lower_triangular = (bw_up == 0);
    d.is_diagonal = (bw_lo == 0 && bw_up == 0);
    d.is_lower_bidiagonal = (bw_up == 0 && bw_lo == 1);
    d.is_upper_bidiagonal = (bw_lo == 0 && bw_up == 1);
    d.is_bidiagonal = d.is_lower_bidiagonal || d.is_upper_bidiagonal;
    // "most specific" reporting: a diagonal or bidiagonal matrix is not also
    // flagged tridiagonal (proper tridiagonal needs both off-diagonals).
    d.is_tridiagonal = (bw_lo <= 1 && bw_up <= 1) && !d.is_diagonal && !d.is_bidiagonal;
    d.is_band = (bw_lo + bw_up) < (R - 1);
    d.is_zero = (nnz == 0);
    // identity / scalar matrix
    if (d.is_diagonal) {
      bool ident = true, scalar_m = true;
      int64_t d0 = at(0, 0);
      for (int64_t i = 0; i < R; ++i) {
        if (at(i, i) != 1) ident = false;
        if (at(i, i) != d0) scalar_m = false;
      }
      d.is_identity = ident;
      d.is_scalar_matrix = scalar_m;
    }
    // symmetry (exact integer)
    bool sym = true, skew = true;
    for (int64_t i = 0; i < R && (sym || skew); ++i)
      for (int64_t j = 0; j < C; ++j) {
        if (at(i, j) != at(j, i)) sym = false;
        if (at(i, j) != -at(j, i)) skew = false;
      }
    d.is_symmetric = sym;
    d.is_skew_symmetric = skew;
    d.is_hermitian = sym;   // real entries: Hermitian == symmetric
    // orthogonal / normal via exact integer products
    const int n = int(R);
    // A^T stored
    std::vector<int64_t> AT(size_t(n * n));
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) AT[i * n + j] = at(j, i);
    auto ATA = imatmul(AT, X, n);      // A^T A
    auto AAT = imatmul(X, AT, n);      // A A^T
    bool orth = true, normal = true;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        if (ATA[i][j] != (i == j ? 1 : 0)) orth = false;
        if (ATA[i][j] != AAT[i][j]) normal = false;
      }
    d.is_orthogonal = orth;
    d.is_normal = normal;
    // trace + determinant
    long double tr = 0.0L;
    for (int i = 0; i < n; ++i) tr += (long double)at(i, i);
    d.trace = tr;
    if (d.is_lower_triangular || d.is_upper_triangular) {
      // exact diagonal product
      __int128 p = 1;
      bool okp = true;
      const __int128 kLimit = (__int128)1000000000000000000LL;
      for (int i = 0; i < n && okp; ++i) {
        __int128 di = at(i, i);
        if (p > kLimit || p < -kLimit) { okp = false; break; }
        p *= di;
      }
      if (okp) {
        d.determinant_exact = true;
        d.determinant = (long double)p;
        d.determinant_str = i128_to_string(p);
      }
    }
    if (!d.determinant_exact) {
      std::vector<std::vector<__int128>> a128(n, std::vector<__int128>(n));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) a128[i][j] = at(i, j);
      __int128 det128;
      if (bareiss_det(a128, n, det128)) {
        d.determinant_exact = true;
        d.determinant = (long double)det128;
        d.determinant_str = i128_to_string(det128);
      } else {
        Mat ad(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j) ad[i][j] = double(at(i, j));
        d.determinant = (long double)lu_det(ad, n);
        d.determinant_str.clear();
      }
    }
  }

  // --- Tier 2: rank / spectrum / definiteness ----------------------------
  if (opts.spectral && std::max(R, C) <= opts.spectral_max_n) {
    Mat A((size_t)R, std::vector<double>((size_t)C, 0.0));
    for (int64_t i = 0; i < R; ++i)
      for (int64_t j = 0; j < C; ++j) A[i][j] = double(at(i, j));
    d.has_rank = true;
    // Exact integer elimination first; the double path is only a fallback for
    // matrices whose minors outgrow the 128-bit guard.
    int exact_rk = 0;
    if (exact_rank(to_i128(X, int(R), int(C)), int(R), int(C), exact_rk)) {
      d.rank = exact_rk;
      d.rank_exact = true;
    } else {
      d.rank = rank_of(A, int(R), int(C), opts.tol);
      d.rank_exact = false;
    }
    d.is_full_rank = (d.rank == std::min(R, C));
    if (d.is_square) {
      d.is_singular = (d.rank < R);
      d.is_invertible = !d.is_singular;
    }

    if (d.is_square) {
      const int n = int(R);
      std::vector<double> ev;
      // Exact Jordan block sizes, parallel to d.spectrum, filled in only for
      // eigenvalues that certified exactly.
      std::vector<std::vector<int>> exact_jordan;
      // `iterative_reliable` marks the paths whose output is a trustworthy
      // approximation of a genuinely real spectrum. The general QR path is not:
      // it silently returns junk for a matrix with complex eigenvalues.
      bool iterative_reliable = true;
      if (d.is_lower_triangular || d.is_upper_triangular) {
        ev.resize(n);
        for (int i = 0; i < n; ++i) ev[i] = double(at(i, i));
        d.eigenvalues_exact = true;
        d.spectral_note = "exact (triangular diagonal)";
      } else if (d.is_symmetric) {
        Mat S(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j) S[i][j] = double(at(i, j));
        ev = jacobi_eigenvalues(S, n, opts.tol);
        d.spectral_note = "real (Jacobi, symmetric)";  // symmetric => real spectrum
      } else {
        Mat G(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j) G[i][j] = double(at(i, j));
        bool conv = true;
        ev = qr_eigenvalues(G, n, opts.tol, conv);
        iterative_reliable = conv;
        d.spectral_note = conv ? "approximate (QR iteration)"
                               : "undetermined (QR did not converge; "
                                 "spectrum may be complex)";
      }
      std::sort(ev.begin(), ev.end());
      d.eigenvalues = ev;
      d.spectral_computed = true;

      // Group into distinct eigenvalues. This is only a first guess at the
      // algebraic multiplicities: unshifted QR converges slowly at a repeated
      // root, so a true double root arrives as a split pair (1.996, 2.004) that
      // no sane tolerance would merge. The exact certification below is what
      // actually decides multiplicity; grouping just seeds the candidate list.
      const double gtol = std::max(opts.tol, 1e-6);
      for (double lam : ev) {
        if (!d.spectrum.empty() &&
            std::fabs(d.spectrum.back().value - lam) <= gtol *
                (1.0 + std::fabs(lam))) {
          d.spectrum.back().algebraic += 1;
        } else {
          Eigenpair ep;
          ep.value = lam;
          d.spectrum.push_back(ep);
        }
      }

      // Integer-eigenvalue search. Certification is exact, so a wrong candidate
      // costs nothing but time — which lets us be generous and probe the
      // neighbours of each rounded estimate. That is what recovers the true
      // double root from a split (1.996, 2.004) pair.
      {
        std::vector<int64_t> cands;
        for (double lam : ev) {
          const int64_t c = (int64_t)std::llround(lam);
          for (int64_t k = c - 1; k <= c + 1; ++k)
            if (std::find(cands.begin(), cands.end(), k) == cands.end())
              cands.push_back(k);
        }
        std::vector<Eigenpair> certified;
        std::vector<std::vector<int>> certified_blocks;
        int total = 0;
        for (int64_t lam : cands) {
          ExactEigen xe;
          if (!certify_integer_eigenvalue(X, n, lam, xe)) continue;
          Eigenpair ep;
          ep.value = (double)lam;
          ep.algebraic = xe.algebraic;
          ep.geometric = xe.geometric;
          ep.exact = true;
          certified.push_back(ep);
          certified_blocks.push_back(xe.jordan);
          total += xe.algebraic;
        }
        // Only adopt the certified set when it accounts for ALL n roots;
        // a partial integer spectrum would silently drop the other eigenvalues.
        if (total == n && !certified.empty()) {
          std::vector<size_t> order(certified.size());
          for (size_t i = 0; i < order.size(); ++i) order[i] = i;
          std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return certified[a].value < certified[b].value;
          });
          d.spectrum.clear();
          exact_jordan.clear();
          for (size_t i : order) {
            d.spectrum.push_back(certified[i]);
            exact_jordan.push_back(certified_blocks[i]);
          }
          d.eigenvalues.clear();
          for (const auto &ep : d.spectrum)
            for (int k = 0; k < ep.algebraic; ++k)
              d.eigenvalues.push_back(ep.value);
          std::sort(d.eigenvalues.begin(), d.eigenvalues.end());
          ev = d.eigenvalues;
          d.eigenvalues_exact = true;
          iterative_reliable = true;
          if (!(d.is_lower_triangular || d.is_upper_triangular))
            d.spectral_note = "exact (integer spectrum certified)";
        }
      }
      const double jtol = std::max(opts.tol, 1e-6);

      // Geometric multiplicity for any eigenvalue the certification did not
      // claim (a non-integer real root): n - rank(A - lambda I), in doubles.
      for (auto &ep : d.spectrum) {
        if (ep.exact) continue;
        Mat B(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            B[i][j] = double(at(i, j)) - (i == j ? ep.value : 0.0);
        ep.geometric = n - rank_of(B, n, n, jtol);
        if (ep.geometric < 1) ep.geometric = 1;
      }

      // HONESTY GATE. An uncertified spectrum is only reported if the iterative
      // path was reliable AND the values reproduce the two invariants we already
      // know exactly: sum == trace and product == determinant. A matrix with a
      // complex spectrum fails this (unshifted QR returns the diagonal of a
      // non-converged iterate, e.g. [0,0] for a 90-degree rotation whose true
      // eigenvalues are +/-i), and we must say "undetermined" rather than
      // publish those numbers as the spectrum.
      d.spectrum_determined = true;
      if (!d.eigenvalues_exact) {
        if (!iterative_reliable) {
          d.spectrum_determined = false;
        } else {
          double sum = 0.0, prod = 1.0;
          for (double lam : ev) { sum += lam; prod *= lam; }
          const double scale = std::max(1.0, std::fabs((double)d.trace));
          if (std::fabs(sum - (double)d.trace) > 1e-6 * scale)
            d.spectrum_determined = false;
          if (d.determinant_exact) {
            const double dscale = std::max(1.0, std::fabs((double)d.determinant));
            if (std::fabs(prod - (double)d.determinant) > 1e-6 * dscale)
              d.spectrum_determined = false;
          }
          if (!d.spectrum_determined)
            d.spectral_note = "undetermined (spectrum inconsistent with trace/"
                              "determinant; may be complex)";
        }
      }
      if (!d.spectrum_determined) {
        // Publish nothing we cannot stand behind. spectral_computed stays true
        // so callers can tell "tier 2 ran and failed" from "tier 2 was skipped".
        d.eigenvalues.clear();
        d.spectrum.clear();
        exact_jordan.clear();
        ev.clear();
      }

      if (d.spectrum_determined) {
        double sr = 0.0;
        bool all_pos = !ev.empty();
        for (double lam : ev) { sr = std::max(sr, std::fabs(lam)); if (lam <= opts.tol) all_pos = false; }
        d.spectral_radius = sr;
        d.all_eigenvalues_positive = all_pos;

        bool diagble = true;
        for (auto &ep : d.spectrum) if (ep.geometric < ep.algebraic) diagble = false;
        d.is_diagonalizable = diagble;
        d.is_defective = !diagble;

        // Jordan normal form: one block per eigenvector. Certified eigenvalues
        // already carry their exact block sizes; the rest fall back to the
        // double rank sequence, and a diagonalizable eigenvalue is trivially
        // all-1 blocks.
        for (size_t s = 0; s < d.spectrum.size(); ++s) {
          const Eigenpair &ep = d.spectrum[s];
          std::vector<int> sizes;
          if (s < exact_jordan.size() && !exact_jordan[s].empty()) {
            sizes = exact_jordan[s];
          } else if (ep.geometric == ep.algebraic) {
            sizes.assign((size_t)ep.algebraic, 1);
          } else {
            sizes = jordan_sizes(A, n, ep.value, ep.algebraic, jtol);
          }
          std::sort(sizes.begin(), sizes.end(), std::greater<int>());
          for (int sz : sizes) d.jordan_blocks.push_back(JordanBlock{ep.value, sz});
        }
      }

      // 2-norm condition number via singular values: sigma_i = sqrt(eig(A^T A)).
      Mat AtA(n, std::vector<double>(n, 0.0));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          double s = 0.0;
          for (int k = 0; k < n; ++k) s += double(at(k, i)) * double(at(k, j));
          AtA[i][j] = s;
        }
      auto ata_ev = jacobi_eigenvalues(AtA, n, opts.tol);
      double smax = 0.0, smin = 0.0;
      bool first = true;
      for (double lam : ata_ev) {
        double s = std::sqrt(std::max(0.0, lam));
        smax = std::max(smax, s);
        if (first || s < smin) { smin = s; first = false; }
      }
      d.has_condition = true;
      if (smin <= opts.tol * std::max(1.0, smax)) {
        d.condition_number = std::numeric_limits<double>::infinity();
        d.condition_status = "singular";
      } else {
        d.condition_number = smax / smin;
        d.condition_status = d.condition_number < 1e3   ? "well_conditioned"
                             : d.condition_number < 1e6 ? "moderately_conditioned"
                                                        : "ill_conditioned";
      }

      // definiteness (meaningful for symmetric matrices)
      if (d.is_symmetric && d.spectrum_determined) {
        bool allpos = true, allneg = true, allnn = true, allnp = true, anyz = false;
        for (double lam : ev) {
          if (lam <= -opts.tol) allpos = false;
          if (lam >= opts.tol) allneg = false;
          if (lam < -opts.tol) allnn = false;
          if (lam > opts.tol) allnp = false;
          if (std::fabs(lam) <= opts.tol) anyz = true;
        }
        d.is_positive_definite = allpos && !anyz;
        d.is_positive_semidefinite = allnn;
        d.is_negative_definite = allneg && !anyz;
        d.is_negative_semidefinite = allnp;
        d.is_indefinite = !allnn && !allnp;
      }

      // special predicates. Nilpotency is a spectral claim (every eigenvalue
      // zero), so it can only be asserted when the spectrum is determined —
      // otherwise an empty `ev` would vacuously "prove" it.
      d.is_nilpotent = d.spectrum_determined && !d.is_zero;
      for (double lam : ev) if (std::fabs(lam) > std::max(opts.tol, 1e-6)) d.is_nilpotent = false;
      auto AA = imatmul(X, X, n);
      bool idem = true, invol = true;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          if (AA[i][j] != (__int128)at(i, j)) idem = false;
          if (AA[i][j] != (__int128)(i == j ? 1 : 0)) invol = false;
        }
      d.is_idempotent = idem;
      d.is_involutory = invol;
    }
  }

  // --- decompositions supported (derived) --------------------------------
  d.supports_qr = true;
  d.supports_svd = true;
  if (d.is_square) {
    d.supports_lu = true;
    d.supports_jordan = true;
    // Undetermined spectrum => we cannot rule eigendecomposition in or out;
    // report the optimistic default rather than a false negative.
    d.supports_eigendecomp = d.spectrum_determined ? d.is_diagonalizable : true;
    d.supports_cholesky = d.is_symmetric && d.is_positive_definite;
  }

  // --- computational / solver hints (derived) ----------------------------
  if (d.is_sparse) d.storage_format_recommended = {"CSR", "CSC", "COO"};
  else d.storage_format_recommended = {"dense_row_major"};

  if (d.is_lower_triangular && !d.is_diagonal) {
    d.preferred_linear_solver = {"forward_substitution"};
    d.preferred_linear_solver.push_back(d.is_sparse ? "sparse_lu" : "lu");
  } else if (d.is_upper_triangular && !d.is_diagonal) {
    d.preferred_linear_solver = {"back_substitution"};
    d.preferred_linear_solver.push_back(d.is_sparse ? "sparse_lu" : "lu");
  } else if (d.is_symmetric && d.is_positive_definite) {
    d.preferred_linear_solver = {"cholesky"};
  } else if (d.is_sparse) {
    d.preferred_linear_solver = {"sparse_lu"};
  } else {
    d.preferred_linear_solver = {"lu", "qr"};
  }

  if (d.is_lower_triangular || d.is_upper_triangular || d.is_diagonal)
    d.preferred_eigen_solver = {"diagonal_read"};
  else if (d.is_symmetric)
    d.preferred_eigen_solver = {"jacobi", "symmetric_qr"};
  else if (d.is_defective)
    d.preferred_eigen_solver = {"jordan_method", "schur_decomposition"};
  else
    d.preferred_eigen_solver = {"qr", "schur_decomposition"};

  // --- numerical (heuristic, approximate) ---------------------------------
  // Only meaningful once a condition number exists; "stable" here means the
  // condition number is not large enough to eat working precision.
  d.is_numerically_stable =
      d.has_condition && (d.condition_status == "well_conditioned" ||
                          d.condition_status == "moderately_conditioned");

  // --- hardware suitability hints (rules over structure) ------------------
  const bool structured =
      d.is_band || d.is_diagonal || d.is_lower_triangular || d.is_upper_triangular;
  // FPGA: integer arithmetic with a predictable access pattern streams well.
  d.fpga_friendly = structured || d.is_sparse;
  // Cache: a narrow band keeps the working set inside a few lines; a dense
  // matrix at least walks contiguous row-major memory.
  d.cache_friendly = (d.is_band && d.total_bandwidth <= 4) || d.is_dense;
  // GPU: wants regularity. Dense is regular; so is structured sparsity.
  // Irregular sparsity is the case that maps badly.
  d.gpu_friendly = d.is_dense || structured;
  // Tensor cores want dense tiles on 8-element boundaries.
  d.tensor_core_friendly = d.is_dense && d.rows % 8 == 0 && d.cols % 8 == 0;

  // --- geometric interpretation (rules) -----------------------------------
  if (d.is_square) {
    // Rotation: an orthogonal map with det +1 preserves lengths and handedness.
    d.represents_rotation =
        d.is_orthogonal && d.determinant_exact && (long double)d.determinant == 1.0L;
    // Projection: idempotent, P^2 = P.
    d.represents_projection = d.is_idempotent;
    // Scaling: the diagonal acts by something other than a pure unit.
    bool scales = false;
    for (int64_t i = 0; i < R; ++i) {
      const int64_t dii = at(i, i);
      if (dii != 1 && dii != 0) { scales = true; break; }
    }
    d.represents_scaling = scales;
    // Shear: off-diagonal coupling that is not a rotation — one coordinate is
    // added into another.
    d.represents_shear = !d.is_diagonal && !d.represents_rotation &&
                         (d.bandwidth_lower > 0 || d.bandwidth_upper > 0);
  }

  // --- dynamic interpretation (rules) -------------------------------------
  for (const auto &b : d.jordan_blocks) {
    if (b.size > 1) {
      d.has_coupled_modes = true;
      char buf[64];
      std::snprintf(buf, sizeof(buf), "lambda %g: jordan_coupled_mode (size %d)",
                    b.eigenvalue, b.size);
      push_unique(d.dynamic_modes, buf);
    }
  }
  if (d.spectrum_determined && !d.has_coupled_modes && !d.spectrum.empty())
    d.dynamic_modes.push_back("all modes independent");

  // --- taxonomy (derived from the declarative rule base) ------------------
  for (const auto &r : taxonomyRules()) {
    bool holds = false;
    if (!evalTaxonomyPredicate(r.antecedent, d, holds)) continue;  // unknown name
    if (!holds) continue;
    switch (r.kind) {
      case RuleKind::Inherits: push_unique(d.inherits, r.consequent); break;
      case RuleKind::Implies:  push_unique(d.implies, r.consequent);  break;
      case RuleKind::Forbids:  push_unique(d.forbids, r.consequent);  break;
    }
  }

  return d;
}

namespace {
std::string join(const std::vector<std::string> &v) {
  std::string s;
  for (const auto &x : v) { if (!s.empty()) s += ", "; s += x; }
  return s.empty() ? "-" : s;
}
std::string num(double v) {
  // integers print without a decimal point
  if (std::fabs(v - std::llround(v)) < 1e-9)
    return std::to_string((long long)std::llround(v));
  char b[32]; std::snprintf(b, sizeof(b), "%.4g", v); return b;
}
void row(std::ostringstream &o, const char *k, const std::string &v) {
  char buf[48]; std::snprintf(buf, sizeof(buf), "  %-26s", k);
  o << buf << v << "\n";
}
void brow(std::ostringstream &o, const char *k, bool v) { row(o, k, v ? "true" : "false"); }
}  // namespace

std::string renderReport(const MatrixDescriptor &d) {
  std::ostringstream o;
  if (!d.ok) { o << "profile error: " << d.error << "\n"; return o.str(); }
  const char *rule = "------------------------------------------------";
  o << "Matrix Profile: " << d.name << "\n" << rule << "\n";

  o << "[basic_id]\n";
  row(o, "type", d.type);
  row(o, "field", d.field);
  row(o, "dtype", d.dtype);

  o << "[dimensions]  exact\n";
  row(o, "shape", std::to_string(d.rows) + " x " + std::to_string(d.cols));
  if (d.is_square) row(o, "order", std::to_string(d.rows));
  brow(o, "is_square", d.is_square);
  brow(o, "is_rectangular", d.is_rectangular);

  o << "[structure]  exact\n";
  brow(o, "is_diagonal", d.is_diagonal);
  brow(o, "is_identity", d.is_identity);
  brow(o, "is_zero_matrix", d.is_zero);
  brow(o, "is_upper_triangular", d.is_upper_triangular);
  brow(o, "is_lower_triangular", d.is_lower_triangular);
  brow(o, "is_bidiagonal", d.is_bidiagonal);
  brow(o, "is_lower_bidiagonal", d.is_lower_bidiagonal);
  brow(o, "is_upper_bidiagonal", d.is_upper_bidiagonal);
  brow(o, "is_tridiagonal", d.is_tridiagonal);
  brow(o, "is_band_matrix", d.is_band);
  row(o, "bandwidth_lower", std::to_string(d.bandwidth_lower));
  row(o, "bandwidth_upper", std::to_string(d.bandwidth_upper));
  brow(o, "is_sparse", d.is_sparse);
  row(o, "non_zero_elements", std::to_string(d.non_zero));
  row(o, "zero_elements", std::to_string(d.zero_count));
  row(o, "sparsity_ratio", std::to_string(d.sparsity_num) + "/" +
                               std::to_string(d.sparsity_den) + "  (" +
                               num(d.sparsity_ratio) + ")");

  o << "[symmetry]  exact\n";
  brow(o, "is_symmetric", d.is_symmetric);
  brow(o, "is_skew_symmetric", d.is_skew_symmetric);
  brow(o, "is_hermitian", d.is_hermitian);
  brow(o, "is_orthogonal", d.is_orthogonal);
  brow(o, "is_normal", d.is_normal);

  o << "[linear_algebra]\n";
  row(o, "trace", num((double)d.trace));
  row(o, "determinant",
      d.determinant_str.empty() ? num((double)d.determinant) + " (approx)"
                                : d.determinant_str);
  if (d.has_rank) {
    row(o, "rank", std::to_string(d.rank) +
                       (d.rank_exact ? "" : "  (approx: exact elimination overflowed)"));
    brow(o, "is_full_rank", d.is_full_rank);
    brow(o, "is_invertible", d.is_invertible);
    brow(o, "is_singular", d.is_singular);
  }

  if (d.spectral_computed) {
    o << "[spectral]  " << d.spectral_note << "\n";
    std::string evs;
    for (double e : d.eigenvalues) { if (!evs.empty()) evs += ", "; evs += num(e); }
    row(o, "eigenvalues", evs.empty() ? "-" : evs);
    std::string mult;
    for (const auto &ep : d.spectrum) {
      if (!mult.empty()) mult += ", ";
      mult += num(ep.value) + " (alg " + std::to_string(ep.algebraic) +
              ", geo " + std::to_string(ep.geometric) + ")";
    }
    row(o, "multiplicity", mult.empty() ? "-" : mult);
    if (!d.spectrum_determined) {
      // Nothing below this line is a claim we can make; say so instead of
      // printing defaults that read as findings.
      row(o, "spectral_radius", "undetermined");
      row(o, "is_diagonalizable", "undetermined");
      row(o, "is_defective", "undetermined");
    } else {
    row(o, "spectral_radius", num(d.spectral_radius));
    brow(o, "all_eigenvalues_positive", d.all_eigenvalues_positive);
    brow(o, "is_diagonalizable", d.is_diagonalizable);
    brow(o, "is_defective", d.is_defective);
    }
    if (!d.jordan_blocks.empty()) {
      std::string jb;
      for (const auto &b : d.jordan_blocks) {
        if (!jb.empty()) jb += ", ";
        jb += num(b.eigenvalue) + "^" + std::to_string(b.size);
      }
      row(o, "jordan_blocks", jb);
    }
    if (d.has_condition) {
      // Tagged at the group level: everything here comes from a double-precision
      // singular-value estimate, so none of it is exact — unlike the groups above.
      o << "[numerical]  approximate\n";
      row(o, "condition_number",
          std::isinf(d.condition_number) ? "inf" : num(d.condition_number));
      row(o, "condition_status", d.condition_status);
      brow(o, "is_numerically_stable", d.is_numerically_stable);
    }
    o << "[positivity]\n";
    if (!d.is_symmetric) {
      // Definiteness is defined via the symmetric part; for a non-symmetric
      // matrix these are not false findings, they simply do not apply.
      row(o, "(definiteness)", "not_applicable (matrix is not symmetric)");
    }
    brow(o, "is_positive_definite", d.is_positive_definite);
    brow(o, "is_positive_semidefinite", d.is_positive_semidefinite);
    brow(o, "is_negative_definite", d.is_negative_definite);
    brow(o, "is_negative_semidefinite", d.is_negative_semidefinite);
    brow(o, "is_indefinite", d.is_indefinite);
    o << "[special_properties]\n";
    brow(o, "is_nilpotent", d.is_nilpotent);
    brow(o, "is_idempotent", d.is_idempotent);
    brow(o, "is_involutory", d.is_involutory);
  }

  o << "[decompositions]\n";
  brow(o, "supports_lu", d.supports_lu);
  brow(o, "supports_qr", d.supports_qr);
  brow(o, "supports_svd", d.supports_svd);
  brow(o, "supports_cholesky", d.supports_cholesky);
  brow(o, "supports_eigendecomp", d.supports_eigendecomp);
  brow(o, "supports_jordan", d.supports_jordan);

  o << "[computational]  rule\n";
  row(o, "storage_format", join(d.storage_format_recommended));
  row(o, "linear_solver", join(d.preferred_linear_solver));
  row(o, "eigen_solver", join(d.preferred_eigen_solver));
  brow(o, "fpga_friendly", d.fpga_friendly);
  brow(o, "cache_friendly", d.cache_friendly);
  brow(o, "gpu_friendly", d.gpu_friendly);
  brow(o, "tensor_core_friendly", d.tensor_core_friendly);

  o << "[geometric_interpretation]  rule\n";
  brow(o, "represents_scaling", d.represents_scaling);
  brow(o, "represents_shear", d.represents_shear);
  brow(o, "represents_rotation", d.represents_rotation);
  brow(o, "represents_projection", d.represents_projection);

  o << "[dynamic_interpretation]  rule\n";
  brow(o, "has_coupled_modes", d.has_coupled_modes);
  row(o, "dynamic_modes", join(d.dynamic_modes));

  o << "[taxonomy]\n";
  row(o, "inherits", join(d.inherits));
  row(o, "implies", join(d.implies));
  row(o, "forbids", join(d.forbids));
  o << rule << "\n";
  return o.str();
}

// ---------------------------------------------------------------------------
// Property access
// ---------------------------------------------------------------------------
namespace {

using interp::Value;

Value vscalar(int64_t v) { return Value::scalar(v); }
Value vbool(bool b) { return Value::scalar(b ? 1 : 0); }

Value vlist(const std::vector<int64_t> &xs) {
  return Value::tensor({(int64_t)xs.size()}, xs);
}

// A field entry produces a Value, or fails with a reason (e.g. a determinant
// too large for int64, or a non-integral eigenvalue).
struct FieldEntry {
  const char *group;
  const char *field;
  bool (*get)(const MatrixDescriptor &, Value &, std::string &);
};

// Convert a double field, refusing to silently truncate.
bool from_double(double v, const char *what, Value &out, std::string &err) {
  if (!std::isfinite(v)) { err = std::string(what) + " is not finite"; return false; }
  if (std::fabs(v - std::llround(v)) > 1e-9) {
    err = std::string(what) + " is not an integer (" + std::to_string(v) +
          "); Caramel values are integer-only";
    return false;
  }
  out = vscalar((int64_t)std::llround(v));
  return true;
}

#define BOOL_FIELD(g, f, expr)                                            \
  {g, f, [](const MatrixDescriptor &d, Value &o, std::string &) -> bool { \
     o = vbool(expr); return true;                                        \
   }}
#define INT_FIELD(g, f, expr)                                             \
  {g, f, [](const MatrixDescriptor &d, Value &o, std::string &) -> bool { \
     o = vscalar((int64_t)(expr)); return true;                           \
   }}

const FieldEntry kFields[] = {
    // --- dimensions -------------------------------------------------------
    INT_FIELD("dimensions", "rows", d.rows),
    INT_FIELD("dimensions", "cols", d.cols),
    INT_FIELD("dimensions", "order", d.is_square ? d.rows : 0),
    BOOL_FIELD("dimensions", "is_square", d.is_square),
    BOOL_FIELD("dimensions", "is_rectangular", d.is_rectangular),
    BOOL_FIELD("dimensions", "is_row_matrix", d.is_row_matrix),
    BOOL_FIELD("dimensions", "is_column_matrix", d.is_column_matrix),
    {"dimensions", "shape",
     [](const MatrixDescriptor &d, Value &o, std::string &) -> bool {
       o = vlist({d.rows, d.cols}); return true;
     }},

    // --- structure --------------------------------------------------------
    BOOL_FIELD("structure", "is_diagonal", d.is_diagonal),
    BOOL_FIELD("structure", "is_scalar_matrix", d.is_scalar_matrix),
    BOOL_FIELD("structure", "is_identity", d.is_identity),
    BOOL_FIELD("structure", "is_zero", d.is_zero),
    BOOL_FIELD("structure", "is_upper_triangular", d.is_upper_triangular),
    BOOL_FIELD("structure", "is_lower_triangular", d.is_lower_triangular),
    BOOL_FIELD("structure", "is_bidiagonal", d.is_bidiagonal),
    BOOL_FIELD("structure", "is_lower_bidiagonal", d.is_lower_bidiagonal),
    BOOL_FIELD("structure", "is_upper_bidiagonal", d.is_upper_bidiagonal),
    BOOL_FIELD("structure", "is_tridiagonal", d.is_tridiagonal),
    BOOL_FIELD("structure", "is_band_matrix", d.is_band),
    BOOL_FIELD("structure", "is_sparse", d.is_sparse),
    BOOL_FIELD("structure", "is_dense", d.is_dense),
    INT_FIELD("structure", "bandwidth_upper", d.bandwidth_upper),
    INT_FIELD("structure", "bandwidth_lower", d.bandwidth_lower),
    INT_FIELD("structure", "total_bandwidth", d.total_bandwidth),
    INT_FIELD("structure", "non_zero_elements", d.non_zero),
    INT_FIELD("structure", "zero_elements", d.zero_count),
    // exact rational: numerator and denominator, never a lossy ratio
    INT_FIELD("structure", "sparsity_num", d.sparsity_num),
    INT_FIELD("structure", "sparsity_den", d.sparsity_den),

    // --- symmetry ---------------------------------------------------------
    BOOL_FIELD("symmetry", "is_symmetric", d.is_symmetric),
    BOOL_FIELD("symmetry", "is_skew_symmetric", d.is_skew_symmetric),
    BOOL_FIELD("symmetry", "is_hermitian", d.is_hermitian),
    BOOL_FIELD("symmetry", "is_orthogonal", d.is_orthogonal),
    BOOL_FIELD("symmetry", "is_normal", d.is_normal),

    // --- linear_algebra ---------------------------------------------------
    {"linear_algebra", "trace",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       return from_double((double)d.trace, "trace", o, e);
     }},
    {"linear_algebra", "determinant",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       if (!d.determinant_exact) {
         e = "determinant is not exact (the exact kernel overflowed); refusing "
             "to return an approximation as an integer";
         return false;
       }
       // The exact value may not fit int64 even though it was computed in 128
       // bits — say so rather than wrap.
       const std::string &s = d.determinant_str;
       try {
         size_t used = 0;
         long long v = std::stoll(s, &used);
         if (used != s.size()) throw std::out_of_range("trailing");
         o = vscalar((int64_t)v);
         return true;
       } catch (...) {
         e = "determinant " + s + " does not fit in a 64-bit integer";
         return false;
       }
     }},
    BOOL_FIELD("linear_algebra", "determinant_exact", d.determinant_exact),
    INT_FIELD("linear_algebra", "rank", d.rank),
    BOOL_FIELD("linear_algebra", "rank_exact", d.rank_exact),
    BOOL_FIELD("linear_algebra", "is_full_rank", d.is_full_rank),
    BOOL_FIELD("linear_algebra", "is_singular", d.is_singular),
    BOOL_FIELD("linear_algebra", "is_invertible", d.is_invertible),

    // --- spectral ---------------------------------------------------------
    {"spectral", "eigenvalues",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       if (!d.spectrum_determined) { e = "spectrum is undetermined"; return false; }
       std::vector<int64_t> xs;
       for (double v : d.eigenvalues) {
         if (std::fabs(v - std::llround(v)) > 1e-9) {
           e = "eigenvalues are not all integers; no integer representation";
           return false;
         }
         xs.push_back((int64_t)std::llround(v));
       }
       o = vlist(xs);
       return true;
     }},
    {"spectral", "distinct_eigenvalues",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       if (!d.spectrum_determined) { e = "spectrum is undetermined"; return false; }
       std::vector<int64_t> xs;
       for (const auto &ep : d.spectrum) {
         if (std::fabs(ep.value - std::llround(ep.value)) > 1e-9) {
           e = "eigenvalues are not all integers; no integer representation";
           return false;
         }
         xs.push_back((int64_t)std::llround(ep.value));
       }
       o = vlist(xs);
       return true;
     }},
    {"spectral", "algebraic_multiplicities",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       if (!d.spectrum_determined) { e = "spectrum is undetermined"; return false; }
       std::vector<int64_t> xs;
       for (const auto &ep : d.spectrum) xs.push_back(ep.algebraic);
       o = vlist(xs);
       return true;
     }},
    {"spectral", "geometric_multiplicities",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       if (!d.spectrum_determined) { e = "spectrum is undetermined"; return false; }
       std::vector<int64_t> xs;
       for (const auto &ep : d.spectrum) xs.push_back(ep.geometric);
       o = vlist(xs);
       return true;
     }},
    {"spectral", "jordan_block_sizes",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       if (!d.spectrum_determined) { e = "spectrum is undetermined"; return false; }
       std::vector<int64_t> xs;
       for (const auto &b : d.jordan_blocks) xs.push_back(b.size);
       o = vlist(xs);
       return true;
     }},
    INT_FIELD("spectral", "jordan_block_count", (int64_t)d.jordan_blocks.size()),
    {"spectral", "spectral_radius",
     [](const MatrixDescriptor &d, Value &o, std::string &e) -> bool {
       if (!d.spectrum_determined) { e = "spectrum is undetermined"; return false; }
       return from_double(d.spectral_radius, "spectral_radius", o, e);
     }},
    BOOL_FIELD("spectral", "all_eigenvalues_positive", d.all_eigenvalues_positive),
    BOOL_FIELD("spectral", "is_diagonalizable", d.is_diagonalizable),
    BOOL_FIELD("spectral", "is_defective", d.is_defective),
    BOOL_FIELD("spectral", "eigenvalues_exact", d.eigenvalues_exact),
    BOOL_FIELD("spectral", "spectrum_determined", d.spectrum_determined),

    // --- positivity -------------------------------------------------------
    BOOL_FIELD("positivity", "is_positive_definite", d.is_positive_definite),
    BOOL_FIELD("positivity", "is_positive_semidefinite", d.is_positive_semidefinite),
    BOOL_FIELD("positivity", "is_negative_definite", d.is_negative_definite),
    BOOL_FIELD("positivity", "is_negative_semidefinite", d.is_negative_semidefinite),
    BOOL_FIELD("positivity", "is_indefinite", d.is_indefinite),

    // --- special_properties ----------------------------------------------
    BOOL_FIELD("special_properties", "is_nilpotent", d.is_nilpotent),
    BOOL_FIELD("special_properties", "is_idempotent", d.is_idempotent),
    BOOL_FIELD("special_properties", "is_involutory", d.is_involutory),

    // --- decompositions ---------------------------------------------------
    BOOL_FIELD("decompositions", "supports_lu", d.supports_lu),
    BOOL_FIELD("decompositions", "supports_qr", d.supports_qr),
    BOOL_FIELD("decompositions", "supports_svd", d.supports_svd),
    BOOL_FIELD("decompositions", "supports_cholesky", d.supports_cholesky),
    BOOL_FIELD("decompositions", "supports_eigendecomposition", d.supports_eigendecomp),
    BOOL_FIELD("decompositions", "supports_jordan_decomposition", d.supports_jordan),

    // --- numerical --------------------------------------------------------
    BOOL_FIELD("numerical", "is_numerically_stable", d.is_numerically_stable),

    // --- computational ----------------------------------------------------
    BOOL_FIELD("computational", "fpga_friendly", d.fpga_friendly),
    BOOL_FIELD("computational", "cache_friendly", d.cache_friendly),
    BOOL_FIELD("computational", "gpu_friendly", d.gpu_friendly),
    BOOL_FIELD("computational", "tensor_core_friendly", d.tensor_core_friendly),

    // --- geometric / dynamic interpretation -------------------------------
    BOOL_FIELD("geometric_interpretation", "represents_scaling", d.represents_scaling),
    BOOL_FIELD("geometric_interpretation", "represents_shear", d.represents_shear),
    BOOL_FIELD("geometric_interpretation", "represents_rotation", d.represents_rotation),
    BOOL_FIELD("geometric_interpretation", "represents_projection", d.represents_projection),
    BOOL_FIELD("dynamic_interpretation", "has_coupled_modes", d.has_coupled_modes),
};

#undef BOOL_FIELD
#undef INT_FIELD

}  // namespace

bool descriptorProperty(const MatrixDescriptor &d,
                        const std::vector<std::string> &path, Value &out,
                        std::string &err) {
  if (!d.ok) { err = d.error; return false; }
  // Drop a leading "properties" segment: `m.properties.structure.is_sparse`
  // and `m.structure.is_sparse` name the same thing.
  std::vector<std::string> p;
  for (const auto &seg : path)
    if (!(p.empty() && seg == "properties")) p.push_back(seg);

  if (p.empty()) { err = "empty property path"; return false; }

  const std::string field = p.back();
  const std::string group = p.size() >= 2 ? p[p.size() - 2] : std::string();

  const FieldEntry *match = nullptr;
  bool field_seen = false;
  for (const auto &f : kFields) {
    if (field != f.field) continue;
    field_seen = true;
    if (group.empty() || group == f.group) { match = &f; break; }
  }
  if (!match) {
    if (field_seen)
      err = "property '" + field + "' exists but not in group '" + group + "'";
    else
      err = "unknown property '" + field + "'";
    return false;
  }
  return match->get(d, out, err);
}

const std::vector<std::string> &descriptorPropertyPaths() {
  static const std::vector<std::string> paths = [] {
    std::vector<std::string> v;
    for (const auto &f : kFields)
      v.push_back(std::string(f.group) + "." + f.field);
    return v;
  }();
  return paths;
}

// ---------------------------------------------------------------------------
// Canonical JSON serialisation
// ---------------------------------------------------------------------------
namespace {

void jstr(std::ostringstream &o, const std::string &s) {
  o << '"';
  for (char c : s) {
    switch (c) {
      case '"':  o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\n': o << "\\n";  break;
      case '\t': o << "\\t";  break;
      default:
        if ((unsigned char)c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", (unsigned)(unsigned char)c);
          o << b;
        } else {
          o << c;
        }
    }
  }
  o << '"';
}

void jkey(std::ostringstream &o, const char *k, bool &first) {
  if (!first) o << ',';
  first = false;
  jstr(o, k);
  o << ':';
}

void jbool(std::ostringstream &o, const char *k, bool v, bool &first) {
  jkey(o, k, first);
  o << (v ? "true" : "false");
}

void jint(std::ostringstream &o, const char *k, long long v, bool &first) {
  jkey(o, k, first);
  o << v;
}

void jtext(std::ostringstream &o, const char *k, const std::string &v, bool &first) {
  jkey(o, k, first);
  jstr(o, v);
}

void jlist(std::ostringstream &o, const char *k,
           const std::vector<std::string> &v, bool &first) {
  jkey(o, k, first);
  o << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) o << ',';
    jstr(o, v[i]);
  }
  o << ']';
}

// A double that is integral prints as an integer, so an exact spectrum
// serialises as [2,2,3,3,5,7,11,13] rather than [2.0,...].
void jnum(std::ostringstream &o, double v) {
  if (std::isfinite(v) && std::fabs(v - std::llround(v)) < 1e-12) {
    o << (long long)std::llround(v);
  } else if (!std::isfinite(v)) {
    o << "null";
  } else {
    char b[40];
    std::snprintf(b, sizeof(b), "%.17g", v);
    o << b;
  }
}

}  // namespace

std::string renderJson(const MatrixDescriptor &d) {
  std::ostringstream o;
  bool top = true;
  o << '{';

  if (!d.ok) {
    jbool(o, "ok", false, top);
    jtext(o, "error", d.error, top);
    o << '}';
    return o.str();
  }
  jbool(o, "ok", true, top);

  jkey(o, "basic_id", top);
  { bool f = true; o << '{';
    jtext(o, "name", d.name, f);
    jtext(o, "type", d.type, f);
    jtext(o, "field", d.field, f);
    jtext(o, "dtype", d.dtype, f);
    o << '}'; }

  jkey(o, "dimensions", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "exact", f);
    jkey(o, "shape", f); o << '[' << d.rows << ',' << d.cols << ']';
    jint(o, "rows", d.rows, f);
    jint(o, "cols", d.cols, f);
    if (d.is_square) jint(o, "order", d.rows, f);
    jbool(o, "is_square", d.is_square, f);
    jbool(o, "is_rectangular", d.is_rectangular, f);
    jbool(o, "is_row_matrix", d.is_row_matrix, f);
    jbool(o, "is_column_matrix", d.is_column_matrix, f);
    o << '}'; }

  jkey(o, "structure", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "exact", f);
    jbool(o, "is_diagonal", d.is_diagonal, f);
    jbool(o, "is_scalar_matrix", d.is_scalar_matrix, f);
    jbool(o, "is_identity", d.is_identity, f);
    jbool(o, "is_zero", d.is_zero, f);
    jbool(o, "is_upper_triangular", d.is_upper_triangular, f);
    jbool(o, "is_lower_triangular", d.is_lower_triangular, f);
    jbool(o, "is_bidiagonal", d.is_bidiagonal, f);
    jbool(o, "is_lower_bidiagonal", d.is_lower_bidiagonal, f);
    jbool(o, "is_upper_bidiagonal", d.is_upper_bidiagonal, f);
    jbool(o, "is_tridiagonal", d.is_tridiagonal, f);
    jbool(o, "is_band_matrix", d.is_band, f);
    jint(o, "bandwidth_upper", d.bandwidth_upper, f);
    jint(o, "bandwidth_lower", d.bandwidth_lower, f);
    jint(o, "total_bandwidth", d.total_bandwidth, f);
    jbool(o, "is_sparse", d.is_sparse, f);
    jbool(o, "is_dense", d.is_dense, f);
    jint(o, "non_zero_elements", d.non_zero, f);
    jint(o, "zero_elements", d.zero_count, f);
    // exact rational, never a lossy float
    jkey(o, "sparsity_ratio", f);
    o << "{\"num\":" << d.sparsity_num << ",\"den\":" << d.sparsity_den << '}';
    o << '}'; }

  jkey(o, "symmetry", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "exact", f);
    jbool(o, "is_symmetric", d.is_symmetric, f);
    jbool(o, "is_skew_symmetric", d.is_skew_symmetric, f);
    jbool(o, "is_hermitian", d.is_hermitian, f);
    jbool(o, "is_orthogonal", d.is_orthogonal, f);
    jbool(o, "is_normal", d.is_normal, f);
    o << '}'; }

  jkey(o, "linear_algebra", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", d.determinant_exact && d.rank_exact ? "exact"
                                                                : "approximate", f);
    jkey(o, "trace", f); jnum(o, (double)d.trace);
    // The exact determinant can exceed double precision, so the authoritative
    // form is the decimal string; `determinant` is the convenience number.
    jkey(o, "determinant", f); jnum(o, (double)d.determinant);
    jbool(o, "determinant_exact", d.determinant_exact, f);
    if (!d.determinant_str.empty())
      jtext(o, "determinant_exact_value", d.determinant_str, f);
    if (d.has_rank) {
      jint(o, "rank", d.rank, f);
      jbool(o, "rank_exact", d.rank_exact, f);
      jbool(o, "is_full_rank", d.is_full_rank, f);
      jbool(o, "is_singular", d.is_singular, f);
      jbool(o, "is_invertible", d.is_invertible, f);
    }
    o << '}'; }

  jkey(o, "spectral", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence",
          !d.spectral_computed  ? "undetermined"
          : !d.spectrum_determined ? "undetermined"
          : d.eigenvalues_exact ? "exact"
                                : "approximate", f);
    jtext(o, "note", d.spectral_note, f);
    jbool(o, "computed", d.spectral_computed, f);
    jbool(o, "determined", d.spectrum_determined, f);
    jkey(o, "eigenvalues", f);
    o << '[';
    for (size_t i = 0; i < d.eigenvalues.size(); ++i) {
      if (i) o << ',';
      jnum(o, d.eigenvalues[i]);
    }
    o << ']';
    jkey(o, "multiplicity", f);
    o << '[';
    for (size_t i = 0; i < d.spectrum.size(); ++i) {
      if (i) o << ',';
      o << "{\"eigenvalue\":"; jnum(o, d.spectrum[i].value);
      o << ",\"algebraic\":" << d.spectrum[i].algebraic
        << ",\"geometric\":" << d.spectrum[i].geometric
        << ",\"exact\":" << (d.spectrum[i].exact ? "true" : "false") << '}';
    }
    o << ']';
    jkey(o, "jordan_blocks", f);
    o << '[';
    for (size_t i = 0; i < d.jordan_blocks.size(); ++i) {
      if (i) o << ',';
      o << "{\"eigenvalue\":"; jnum(o, d.jordan_blocks[i].eigenvalue);
      o << ",\"size\":" << d.jordan_blocks[i].size << '}';
    }
    o << ']';
    if (d.spectrum_determined) {
      jkey(o, "spectral_radius", f); jnum(o, d.spectral_radius);
      jbool(o, "all_eigenvalues_positive", d.all_eigenvalues_positive, f);
      jbool(o, "is_diagonalizable", d.is_diagonalizable, f);
      jbool(o, "is_defective", d.is_defective, f);
    }
    o << '}'; }

  jkey(o, "positivity", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", d.is_symmetric ? "exact" : "not_applicable", f);
    jbool(o, "is_positive_definite", d.is_positive_definite, f);
    jbool(o, "is_positive_semidefinite", d.is_positive_semidefinite, f);
    jbool(o, "is_negative_definite", d.is_negative_definite, f);
    jbool(o, "is_negative_semidefinite", d.is_negative_semidefinite, f);
    jbool(o, "is_indefinite", d.is_indefinite, f);
    o << '}'; }

  jkey(o, "special_properties", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", d.spectrum_determined ? "exact" : "undetermined", f);
    jbool(o, "is_nilpotent", d.is_nilpotent, f);
    jbool(o, "is_idempotent", d.is_idempotent, f);
    jbool(o, "is_involutory", d.is_involutory, f);
    o << '}'; }

  jkey(o, "decompositions", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "rule", f);
    jbool(o, "supports_lu", d.supports_lu, f);
    jbool(o, "supports_qr", d.supports_qr, f);
    jbool(o, "supports_svd", d.supports_svd, f);
    jbool(o, "supports_cholesky", d.supports_cholesky, f);
    jbool(o, "supports_eigendecomposition", d.supports_eigendecomp, f);
    jbool(o, "supports_jordan_decomposition", d.supports_jordan, f);
    o << '}'; }

  jkey(o, "numerical", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "approximate", f);
    if (d.has_condition) {
      jkey(o, "condition_number", f); jnum(o, d.condition_number);
      jtext(o, "condition_status", d.condition_status, f);
    }
    jbool(o, "is_numerically_stable", d.is_numerically_stable, f);
    o << '}'; }

  jkey(o, "computational", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "rule", f);
    jlist(o, "storage_format_recommended", d.storage_format_recommended, f);
    jbool(o, "fpga_friendly", d.fpga_friendly, f);
    jbool(o, "cache_friendly", d.cache_friendly, f);
    jbool(o, "gpu_friendly", d.gpu_friendly, f);
    jbool(o, "tensor_core_friendly", d.tensor_core_friendly, f);
    o << '}'; }

  jkey(o, "recommended_solvers", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "rule", f);
    jlist(o, "linear_solver", d.preferred_linear_solver, f);
    jlist(o, "eigen_solver", d.preferred_eigen_solver, f);
    o << '}'; }

  jkey(o, "geometric_interpretation", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "rule", f);
    jbool(o, "represents_scaling", d.represents_scaling, f);
    jbool(o, "represents_shear", d.represents_shear, f);
    jbool(o, "represents_rotation", d.represents_rotation, f);
    jbool(o, "represents_projection", d.represents_projection, f);
    o << '}'; }

  jkey(o, "dynamic_interpretation", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "rule", f);
    jbool(o, "has_coupled_modes", d.has_coupled_modes, f);
    jlist(o, "dynamic_modes", d.dynamic_modes, f);
    o << '}'; }

  jkey(o, "taxonomy", top);
  { bool f = true; o << '{';
    jtext(o, "_confidence", "rule", f);
    jlist(o, "inherits", d.inherits, f);
    jlist(o, "implies", d.implies, f);
    jlist(o, "forbids", d.forbids, f);
    o << '}'; }

  o << '}';
  return o.str();
}

}  // namespace caramel::analysis
