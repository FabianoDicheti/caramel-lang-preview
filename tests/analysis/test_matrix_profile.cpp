// ============================================================================
// Caramel Language - Matrix profiler tests
// ----------------------------------------------------------------------------
// Fixture is the `matriz_a` matrix from the language design notes (../todo.md):
// a lower-bidiagonal, banded, sparse, defective 8x8 with two 2x2 Jordan blocks.
// ============================================================================
#include "caramel/analysis/matrix_profile.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace caramel::analysis;
using caramel::interp::Value;

static int g_failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                               \
    }                                                             \
  } while (0)

static Value mat(std::vector<int64_t> dims, std::vector<int64_t> data) {
  return Value::tensor(std::move(dims), std::move(data));
}

static void test_flagship_matriz_a() {
  // 8x8 lower-bidiagonal with diagonal 2,2,3,3,5,7,11,13 and unit subdiagonal
  // entries at (1,0) and (3,2).
  Value m = mat({8, 8}, {
      2, 0, 0, 0, 0, 0, 0, 0,
      1, 2, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0,
      0, 0, 1, 3, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0,
      0, 0, 0, 0, 0, 7, 0, 0,
      0, 0, 0, 0, 0, 0, 11, 0,
      0, 0, 0, 0, 0, 0, 0, 13});
  auto d = profile(m, "matriz_a");
  CHECK(d.ok);
  // structure (Tier 1, exact)
  CHECK(d.is_square);
  CHECK(d.is_lower_triangular);
  CHECK(!d.is_upper_triangular);
  CHECK(d.is_lower_bidiagonal);
  CHECK(d.is_bidiagonal);
  CHECK(!d.is_tridiagonal);           // most-specific reporting
  CHECK(d.is_band);
  CHECK(d.bandwidth_lower == 1);
  CHECK(d.bandwidth_upper == 0);
  CHECK(d.is_sparse);
  CHECK(d.non_zero == 10);
  CHECK(d.zero_count == 54);
  CHECK((long long)d.trace == 46);
  CHECK(d.determinant_exact);
  CHECK(d.determinant_str == "180180");   // 2*2*3*3*5*7*11*13
  // symmetry
  CHECK(!d.is_symmetric);
  CHECK(!d.is_orthogonal);
  // rank / spectrum (Tier 2)
  CHECK(d.has_rank && d.rank == 8);
  CHECK(d.is_full_rank && d.is_invertible && !d.is_singular);
  CHECK(d.spectral_computed && d.eigenvalues_exact);
  CHECK(d.eigenvalues.size() == 8);
  const double want[8] = {2, 2, 3, 3, 5, 7, 11, 13};
  for (int i = 0; i < 8; ++i) CHECK(std::fabs(d.eigenvalues[i] - want[i]) < 1e-9);
  CHECK((long long)d.spectral_radius == 13);
  CHECK(d.is_defective && !d.is_diagonalizable);
  // eigenvalue 2 and 3 each have algebraic 2, geometric 1 (Jordan blocks)
  int checked = 0;
  for (const auto &ep : d.spectrum) {
    if (std::fabs(ep.value - 2) < 1e-6 || std::fabs(ep.value - 3) < 1e-6) {
      CHECK(ep.algebraic == 2 && ep.geometric == 1);
      ++checked;
    }
  }
  CHECK(checked == 2);
  // Jordan form: 2 and 3 each carry a single 2x2 block; 5,7,11,13 are 1x1.
  // Six blocks total, sizes summing to the order.
  CHECK(d.jordan_blocks.size() == 6);
  int total = 0, twos = 0;
  for (const auto &b : d.jordan_blocks) {
    total += b.size;
    if (b.size == 2) {
      ++twos;
      CHECK(std::fabs(b.eigenvalue - 2) < 1e-6 ||
            std::fabs(b.eigenvalue - 3) < 1e-6);
    }
  }
  CHECK(total == 8);
  CHECK(twos == 2);
  // conditioning: sigma ratio 13/2 here, comfortably well conditioned
  CHECK(d.has_condition);
  CHECK(d.condition_status == "well_conditioned");
  // sparsity reported as an exact fraction, not just a double
  CHECK(d.sparsity_num == 54 && d.sparsity_den == 64);
  // derived
  CHECK(!d.supports_cholesky);
  CHECK(d.supports_jordan);
}

// A diagonalizable matrix must produce only 1x1 Jordan blocks, and a singular
// one must report condition_status "singular" rather than a finite number.
static void test_jordan_and_conditioning_edges() {
  Value diagble = mat({3, 3}, {5, 0, 0, 0, 5, 0, 0, 0, 9});
  auto d = profile(diagble, "d");
  CHECK(d.is_diagonalizable && !d.is_defective);
  CHECK(d.jordan_blocks.size() == 3);
  for (const auto &b : d.jordan_blocks) CHECK(b.size == 1);

  Value sing = mat({2, 2}, {1, 0, 0, 0});
  auto s = profile(sing, "s");
  CHECK(s.has_condition);
  CHECK(s.condition_status == "singular");

  // A single 3x3 Jordan block: one block of size 3, geometric multiplicity 1.
  Value j3 = mat({3, 3}, {4, 1, 0, 0, 4, 1, 0, 0, 4});
  auto j = profile(j3, "j3");
  CHECK(j.is_defective);
  CHECK(j.jordan_blocks.size() == 1);
  CHECK(j.jordan_blocks[0].size == 3);
  CHECK(std::fabs(j.jordan_blocks[0].eigenvalue - 4) < 1e-6);
}

static void test_symmetric_positive_definite() {
  Value m = mat({2, 2}, {2, 1, 1, 2});   // eigenvalues 1 and 3
  auto d = profile(m, "s");
  CHECK(d.is_symmetric);
  CHECK((long long)d.determinant == 3);
  CHECK(d.spectral_computed);
  CHECK(std::fabs(d.eigenvalues.front() - 1) < 1e-6);
  CHECK(std::fabs(d.eigenvalues.back() - 3) < 1e-6);
  CHECK(d.is_positive_definite);
  CHECK(!d.is_indefinite);
  CHECK(d.supports_cholesky);
  CHECK(d.is_diagonalizable);
}

static void test_identity_and_diagonal() {
  Value id = mat({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
  auto d = profile(id, "I");
  CHECK(d.is_identity && d.is_diagonal && d.is_symmetric);
  CHECK((long long)d.determinant == 1);
  CHECK(d.rank == 3 && d.is_invertible);
}

static void test_singular_idempotent() {
  Value p = mat({2, 2}, {1, 0, 0, 0});
  auto d = profile(p, "p");
  CHECK((long long)d.determinant == 0);
  CHECK(d.is_singular && !d.is_invertible);
  CHECK(d.rank == 1);
  CHECK(d.is_idempotent);
}

// A DEFECTIVE matrix that is not triangular, so the exact answer is reachable
// only through integer certification. Unshifted QR splits the double root 2
// into (1.996, 2.004); before certification existed this matrix was reported as
// two distinct eigenvalues and "diagonalizable", the exact opposite of true.
static void test_non_triangular_defective_certifies() {
  Value m = mat({2, 2}, {3, 1, -1, 1});   // charpoly (x-2)^2
  auto d = profile(m, "m");
  CHECK(d.spectrum_determined);
  CHECK(d.eigenvalues_exact);
  CHECK(d.spectrum.size() == 1);
  CHECK(d.spectrum[0].value == 2.0);
  CHECK(d.spectrum[0].algebraic == 2);
  CHECK(d.spectrum[0].geometric == 1);
  CHECK(d.spectrum[0].exact);
  CHECK(d.is_defective && !d.is_diagonalizable);
  CHECK(d.jordan_blocks.size() == 1 && d.jordan_blocks[0].size == 2);
  CHECK(d.rank_exact && d.rank == 2);
}

// A symmetric integer matrix whose spectrum is integral: Jacobi gets it to
// within a tolerance, certification promotes it to exact.
static void test_symmetric_spectrum_certifies_exact() {
  Value m = mat({2, 2}, {2, 1, 1, 2});    // eigenvalues 1 and 3
  auto d = profile(m, "m");
  CHECK(d.eigenvalues_exact);
  CHECK(d.eigenvalues.size() == 2);
  CHECK(d.eigenvalues[0] == 1.0 && d.eigenvalues[1] == 3.0);
  CHECK(d.is_diagonalizable && !d.is_defective);
}

// A COMPLEX spectrum must be refused, not faked. The 90-degree rotation has
// eigenvalues +/-i; the real QR path cannot represent them and returns the
// diagonal of a non-converged iterate ([0,0]). Publishing that would contradict
// the exactly-known determinant of 1, so the profiler must report nothing.
static void test_complex_spectrum_reported_undetermined() {
  Value m = mat({2, 2}, {0, -1, 1, 0});
  auto d = profile(m, "m");
  CHECK(d.spectral_computed);        // tier 2 did run
  CHECK(!d.spectrum_determined);     // but produced no usable spectrum
  CHECK(d.eigenvalues.empty());
  CHECK(d.spectrum.empty());
  CHECK(d.jordan_blocks.empty());
  CHECK(!d.eigenvalues_exact);
  CHECK(!d.is_defective && !d.is_diagonalizable);   // neither is claimed
  CHECK(!d.is_nilpotent);            // must not be vacuously true on empty ev
  // the exactly-computable facts are still reported
  CHECK(d.determinant_exact && (long long)d.determinant == 1);
  CHECK(d.rank_exact && d.rank == 2);
}

// The zero matrix is nilpotent, and its spectrum {0} is determined.
static void test_zero_matrix_nilpotent() {
  Value m = mat({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto d = profile(m, "m");
  CHECK(d.is_zero);
  CHECK(d.rank_exact && d.rank == 0);
  CHECK((long long)d.determinant == 0);
}

// The rule base is data, so a typo in an antecedent would silently disable a
// rule rather than fail to compile. Check every antecedent resolves, and that
// the fixture reproduces the implies/forbids edges the design notes call out.
static void test_taxonomy_rule_base_integrity() {
  MatrixDescriptor dummy;
  for (const auto &r : taxonomyRules()) {
    bool held = false;
    if (!evalTaxonomyPredicate(r.antecedent, dummy, held)) {
      std::printf("FAIL unknown antecedent in rule base: %s\n", r.antecedent);
      ++g_failures;
    }
    CHECK(r.consequent != nullptr && r.consequent[0] != '\0');
  }
  CHECK(!taxonomyPredicateNames().empty());
}

static bool has(const std::vector<std::string> &v, const char *s) {
  for (const auto &x : v) if (x == s) return true;
  return false;
}

static void test_taxonomy_edges_on_fixture() {
  Value m = mat({8, 8}, {
      2, 0, 0, 0, 0, 0, 0, 0,
      1, 2, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0,
      0, 0, 1, 3, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0,
      0, 0, 0, 0, 0, 7, 0, 0,
      0, 0, 0, 0, 0, 0, 11, 0,
      0, 0, 0, 0, 0, 0, 0, 13});
  auto d = profile(m, "matriz_a");
  // inherits, per the design notes' taxonomy section
  for (const char *t : {"square_matrix", "lower_triangular_matrix", "band_matrix",
                        "bidiagonal_matrix", "sparse_matrix", "full_rank_matrix",
                        "invertible_matrix", "defective_matrix"})
    CHECK(has(d.inherits, t));
  // implies
  CHECK(has(d.implies, "eigenvalues_equal_diagonal"));
  CHECK(has(d.implies, "determinant_equals_diagonal_product"));
  CHECK(has(d.implies, "invertible"));
  // forbids
  CHECK(has(d.forbids, "diagonalizable"));
  CHECK(has(d.forbids, "symmetric"));
  CHECK(has(d.forbids, "spd"));
  // an orthogonal matrix must forbid defective (general rule, not fixture-driven)
  Value id = mat({2, 2}, {1, 0, 0, 1});
  auto o = profile(id, "I");
  CHECK(o.is_orthogonal);
  CHECK(has(o.forbids, "defective"));
}

static void test_non_matrix_rejected() {
  Value v = mat({3}, {1, 2, 3});   // rank-1: a vector, not a matrix
  auto d = profile(v, "v");
  CHECK(!d.ok);
  CHECK(!d.error.empty());
}

int main() {
  test_flagship_matriz_a();
  test_jordan_and_conditioning_edges();
  test_symmetric_positive_definite();
  test_identity_and_diagonal();
  test_singular_idempotent();
  test_non_triangular_defective_certifies();
  test_symmetric_spectrum_certifies_exact();
  test_complex_spectrum_reported_undetermined();
  test_zero_matrix_nilpotent();
  test_taxonomy_rule_base_integrity();
  test_taxonomy_edges_on_fixture();
  test_non_matrix_rejected();
  if (g_failures == 0) {
    std::printf("OK: all matrix-profile tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
