// ============================================================================
// Caramel Language - Matrix algebraic profiler (host-side introspection)
// ----------------------------------------------------------------------------
// A single entry point that computes the full algebraic property set of a
// matrix: structure, symmetry, rank/determinant/trace, spectrum, definiteness,
// supported decompositions, and derived taxonomy (inherits / implies / forbids)
// plus recommended storage formats and solvers.
//
// DESIGN: this is TIER B in the two-tier model - it runs OFF the integer
// execution hot path (never inside a dataflow flow), so it may use double or
// wide-integer math internally. The runtime interp::Value stays integer-only;
// the profile is introspection output, not a first-class runtime value.
//
// The analysis is TIERED:
//   * Tier 1 (always): exact, cheap O(n^2) integer structural facts - shape,
//     triangular/band/bidiagonal structure, symmetry, sparsity, trace, and an
//     exact determinant (fraction-free Bareiss over 128-bit integers).
//   * Tier 2 (opt-in, ProfileOptions::spectral, guarded by spectral_max_n):
//     rank, invertibility, eigenvalues/multiplicity, definiteness,
//     diagonalizability, special-matrix predicates, decomposition support.
//
// This MatrixDescriptor is the shared schema the future "synthesis from
// characteristics" engine will invert (build a matrix matching a descriptor).
// ============================================================================
#ifndef CARAMEL_ANALYSIS_MATRIX_PROFILE_H
#define CARAMEL_ANALYSIS_MATRIX_PROFILE_H

#include <cstdint>
#include <string>
#include <vector>

#include "caramel/interp/value.h"

namespace caramel::analysis {

// One eigenvalue with its algebraic (multiplicity in the char. polynomial) and
// geometric (dim of eigenspace) multiplicities. `value` is real; complex
// spectra are reported approximately with a note (see spectral_note).
//
// `exact` means the value was CERTIFIED, not merely computed: it is an integer
// lambda for which det(A - lambda I) == 0 was confirmed in exact integer
// arithmetic, and whose geometric multiplicity came from an exact integer rank.
// An approximate eigenvalue from the QR/Jacobi path always has exact == false,
// so the profiler never presents an estimate as a proof.
struct Eigenpair {
  double value = 0.0;
  int algebraic = 1;
  int geometric = 1;
  bool exact = false;
};

// One Jordan block of the Jordan normal form: `size` copies of `eigenvalue`
// on the diagonal with 1s on the superdiagonal. A matrix is diagonalizable
// iff every block has size 1; the blocks are what the synthesis engine needs
// to rebuild a defective matrix from its spectrum.
struct JordanBlock {
  double eigenvalue = 0.0;
  int size = 1;
};

struct MatrixDescriptor {
  // --- validity ------------------------------------------------------------
  bool ok = false;         // false when the input is not a rank-2 matrix
  std::string error;       // why ok == false

  // --- basic_id ------------------------------------------------------------
  std::string name;
  std::string type = "matrix";
  std::string field = "real";
  std::string dtype = "int";

  // --- dimensions ----------------------------------------------------------
  int64_t rows = 0, cols = 0;
  bool is_square = false, is_rectangular = false;
  bool is_row_matrix = false, is_column_matrix = false;

  // --- structure (Tier 1, exact) ------------------------------------------
  bool is_diagonal = false, is_scalar_matrix = false;
  bool is_identity = false, is_zero = false;
  bool is_upper_triangular = false, is_lower_triangular = false;
  bool is_bidiagonal = false, is_lower_bidiagonal = false, is_upper_bidiagonal = false;
  bool is_tridiagonal = false, is_band = false;
  int64_t bandwidth_upper = 0, bandwidth_lower = 0, total_bandwidth = 0;
  bool is_sparse = false, is_dense = false;
  int64_t non_zero = 0, zero_count = 0;
  // Reported as an exact fraction (zero_count / rows*cols) to stay
  // integer-honest; sparsity_ratio is the convenience decimal rendering.
  int64_t sparsity_num = 0, sparsity_den = 0;
  double sparsity_ratio = 0.0;

  // --- symmetry (Tier 1, exact) -------------------------------------------
  bool is_symmetric = false, is_skew_symmetric = false;
  bool is_orthogonal = false, is_normal = false;
  // Caramel values are real integers, so Hermitian collapses to symmetric.
  // Carried explicitly because the descriptor is the schema a complex-capable
  // successor would extend, and a missing field reads as "false" not "n/a".
  bool is_hermitian = false;

  // --- linear_algebra ------------------------------------------------------
  long double trace = 0.0L;                 // exact for integer input
  bool determinant_exact = false;           // Bareiss succeeded (no overflow)
  long double determinant = 0.0L;           // value (exact or approximate)
  std::string determinant_str;              // exact decimal when available
  bool has_rank = false;                    // Tier 2 computed it
  bool rank_exact = false;                  // exact integer elimination (not float)
  int64_t rank = 0;
  bool is_full_rank = false, is_singular = false, is_invertible = false;

  // --- spectral (Tier 2) ---------------------------------------------------
  bool spectral_computed = false;           // tier 2 ran (vs. was skipped)
  // False when tier 2 ran but could not produce a spectrum it stands behind —
  // typically a complex spectrum, which the real-valued QR path cannot express.
  // The eigenvalue/multiplicity/Jordan fields are then EMPTY rather than
  // populated with junk, and is_diagonalizable/is_defective are not claims.
  bool spectrum_determined = false;
  std::string spectral_note;                // e.g. "exact (triangular)", "approximate (QR)"
  std::vector<Eigenpair> spectrum;          // distinct eigenvalues + multiplicities
  std::vector<double> eigenvalues;          // full list (with multiplicity), sorted
  bool eigenvalues_exact = false;
  double spectral_radius = 0.0;
  bool all_eigenvalues_positive = false;
  bool is_diagonalizable = false, is_defective = false;
  // Jordan normal form block structure, one entry per block, ordered by
  // eigenvalue then descending size. Derived from the rank sequence
  // rank((A-lambda I)^k), so it is only as exact as `eigenvalues_exact`.
  std::vector<JordanBlock> jordan_blocks;

  // --- conditioning (Tier 2) ----------------------------------------------
  bool has_condition = false;
  double condition_number = 0.0;   // 2-norm cond = sigma_max / sigma_min
  // "well_conditioned" | "moderately_conditioned" | "ill_conditioned" |
  // "singular" (sigma_min ~ 0, cond is reported as infinity)
  std::string condition_status;

  // --- positivity (symmetric matrices) ------------------------------------
  bool is_positive_definite = false, is_positive_semidefinite = false;
  bool is_negative_definite = false, is_negative_semidefinite = false;
  bool is_indefinite = false;

  // --- special properties --------------------------------------------------
  bool is_nilpotent = false, is_idempotent = false, is_involutory = false;

  // --- decompositions supported (derived) ---------------------------------
  bool supports_lu = false, supports_qr = false, supports_svd = false;
  bool supports_cholesky = false, supports_eigendecomp = false, supports_jordan = false;

  // --- numerical (heuristic, APPROXIMATE) ---------------------------------
  // Never exact: derived from the double-precision condition number.
  bool is_numerically_stable = false;

  // --- computational / solver hints (derived) -----------------------------
  std::vector<std::string> storage_format_recommended;
  std::vector<std::string> preferred_linear_solver;
  std::vector<std::string> preferred_eigen_solver;
  // Hardware-suitability hints. These are RULES over the structure, not
  // measurements — a hint being true means "the shape suits this target",
  // never "this was benchmarked".
  bool fpga_friendly = false, cache_friendly = false;
  bool gpu_friendly = false, tensor_core_friendly = false;

  // --- geometric interpretation (derived rules) ---------------------------
  // What the matrix does when read as a linear map.
  bool represents_scaling = false, represents_shear = false;
  bool represents_rotation = false, represents_projection = false;

  // --- dynamic interpretation (derived rules) -----------------------------
  // What the matrix does when read as the generator of a discrete/continuous
  // dynamical system. A Jordan block larger than 1x1 means modes that are not
  // independent: the classic coupled (secular / polynomial-growth) mode.
  bool has_coupled_modes = false;
  std::vector<std::string> dynamic_modes;

  // --- taxonomy (derived) --------------------------------------------------
  std::vector<std::string> inherits;
  std::vector<std::string> implies;
  std::vector<std::string> forbids;
};

struct ProfileOptions {
  bool spectral = true;          // compute Tier 2 (rank/eigen/definiteness/...)
  int64_t spectral_max_n = 256;  // skip the O(n^3) spectral work above this size
  double tol = 1e-9;             // numeric tolerance for rank / eigen grouping
};

// ---------------------------------------------------------------------------
// Taxonomy rule base
// ---------------------------------------------------------------------------
// The implies/forbids lattice as DATA rather than control flow. The profiler
// reads it forward (given a matrix, derive its tags); the synthesis engine is
// meant to read it backward (given a wanted tag set, reject specs that violate
// a forbids edge). Both directions must consult the same table, which is only
// possible if the table is a table.
//
// Every rule has exactly one antecedent — a named predicate over the
// descriptor, optionally negated with a leading '!'. Conjunctions live in the
// predicate vocabulary (e.g. "strictly_lower_triangular") rather than in the
// rules, so that a rule stays invertible.
enum class RuleKind { Inherits, Implies, Forbids };

struct TaxonomyRule {
  const char *antecedent;   // predicate name, optionally prefixed with '!'
  RuleKind kind;
  const char *consequent;   // tag attached when the antecedent holds
};

// The rule base, in application order.
const std::vector<TaxonomyRule> &taxonomyRules();

// Evaluate a named predicate (leading '!' negates) against a descriptor.
// Returns false and leaves `out` untouched if the name is not in the
// vocabulary — callers should treat that as a rule-base bug, not a false value.
bool evalTaxonomyPredicate(const std::string &name, const MatrixDescriptor &d,
                           bool &out);

// Every predicate name the rule base may reference.
const std::vector<std::string> &taxonomyPredicateNames();

// ---------------------------------------------------------------------------
// Property access
// ---------------------------------------------------------------------------
// Resolve a descriptor field by dotted path so a profile result can feed
// ordinary program logic (`m.properties.linear_algebra.determinant`), not just
// a printed report.
//
// The path may be fully qualified ("linear_algebra.determinant") or just the
// field ("determinant"), in which case the groups are searched in order. A
// leading "properties" segment is accepted and skipped.
//
// Caramel is integer-only, so the mapping is:
//   integer / double field -> rank-0 scalar (a non-integral double is rejected
//                             rather than silently truncated)
//   boolean field          -> scalar 0 or 1
//   list field             -> rank-1 tensor
//   string field           -> rejected: no integer representation
// Returns false with `err` set on an unknown path or an unrepresentable field.
bool descriptorProperty(const MatrixDescriptor &d,
                        const std::vector<std::string> &path,
                        interp::Value &out, std::string &err);

// Every dotted path descriptorProperty accepts, for diagnostics and docs.
const std::vector<std::string> &descriptorPropertyPaths();

// Profile `m` (must be a rank-2 tensor). `name` fills basic_id.name. On a
// non-matrix input the returned descriptor has ok == false and a set error.
MatrixDescriptor profile(const interp::Value &m, const std::string &name,
                         const ProfileOptions &opts = ProfileOptions{});

// Render a descriptor as an aligned, sectioned text report.
std::string renderReport(const MatrixDescriptor &d);

// Serialise a descriptor as canonical JSON: the schema grouping of the design
// notes, keys in a fixed order, no incidental whitespace, so two runs of the
// same matrix are byte-identical and diffable. This is the machine-readable
// form of the shared vocabulary — what a synthesis engine would consume as its
// input spec. (proto/json_lite.h is a parser for flat control-plane objects
// only; it has no writer and no nesting, hence the local emitter.)
//
// Exactness is carried in the payload, not implied: groups that can be
// approximate ship a sibling "_confidence" key with one of
// "exact" | "approximate" | "rule" | "undetermined" | "not_applicable".
std::string renderJson(const MatrixDescriptor &d);

}  // namespace caramel::analysis

#endif  // CARAMEL_ANALYSIS_MATRIX_PROFILE_H
