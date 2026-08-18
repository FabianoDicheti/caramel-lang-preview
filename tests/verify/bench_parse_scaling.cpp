// ============================================================================
// Objectives verification - O(n) parsing scaling
// ----------------------------------------------------------------------------
// Objective: O(n) parsing  (LANGUAGE_SPEC_RPN.md:64)
// Times lex+parse at sizes N, 2N, 4N, 8N and checks the per-doubling ratio is
// near 2 (linear), not ~4 (quadratic). Emits RESULT lines.
// ============================================================================
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace caramel::parse;

static std::string make_program(int n) {
  std::string s =
      "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n"
      "calc::lambda_flow big(a, b) {\n  a b add v0 =\n";
  for (int i = 1; i < n; ++i)
    s += "  v" + std::to_string(i - 1) + " a add v" + std::to_string(i) + " =\n";
  s += "} return v" + std::to_string(n - 1) + ";\n";
  return s;
}

static double time_parse(int n) {
  std::string src = make_program(n);
  // best-of-3 to reduce noise
  double best = 1e18;
  for (int r = 0; r < 3; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    Lexer lx(src);
    Parser p(lx.tokenize());
    auto prog = p.parse();
    auto t1 = std::chrono::steady_clock::now();
    (void)prog;
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

int main() {
  const int base = 1000;
  std::vector<int> sizes = {base, 2 * base, 4 * base, 8 * base};
  std::vector<double> t;
  for (int n : sizes) t.push_back(time_parse(n));

  for (size_t i = 0; i < sizes.size(); ++i)
    std::printf("RESULT parse_time_n%d_ms MEASURE %.3f\n", sizes[i], t[i]);

  // ratios per doubling; linear ~2.0, quadratic ~4.0. Allow a generous band for
  // timer noise on small absolute times.
  bool linear = true;
  double worst = 0.0;
  for (size_t i = 1; i < t.size(); ++i) {
    double ratio = (t[i - 1] > 1e-6) ? t[i] / t[i - 1] : 2.0;
    worst = std::max(worst, ratio);
    std::printf("RESULT parse_ratio_x%zu MEASURE %.3f\n", i, ratio);
    if (ratio > 3.0) linear = false;  // clearly super-linear if > 3x per doubling
  }
  std::printf("RESULT parse_linear_scaling %s worst_doubling_ratio=%.2f target<3.0\n",
              linear ? "PASS" : "FAIL", worst);
  return linear ? 0 : 1;
}
