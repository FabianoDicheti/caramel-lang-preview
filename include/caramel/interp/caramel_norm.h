/*
 * caramel_norm.h - shared bit-exact fixed-point normalization kernels.
 *
 * Hand-authored, byte-identical in bark/ and caramel_lang/ (like
 * caramel_activation.h). Pure integer (int64), C and C++ compatible, NO 128-bit
 * division (so it links in the freestanding -mno-sse worker without libgcc).
 * Uses an exact integer isqrt instead of a LUT: deterministic and reproducible
 * across both repos, so remote == local byte-exact (--verify). Semantics are
 * normative: PROTOCOL_SPEC.md 6.5 "Normalization".
 *
 * layernorm over one axis-group of N register elements (real = i / 10^q):
 *   mean  = (1/N) Σ x_i
 *   var   = (1/N) Σ (x_i-mean)^2
 *   out_i = (x_i - mean) / sqrt(var)            [affine gamma/beta = 1/0 in v1]
 * computed entirely in integers with a 10-bit fractional scale on the sqrt.
 *
 * RANGE (v1, int64-safe): requires N*Σ(N*x_i-Σx)^2 <= 2^43 (roughly: feature
 * dim <= ~50 at |x|~1000, more for smaller values). Larger dims need a 128-bit
 * path — tracked in x86_063. Both sides compute identically regardless, so a
 * job outside the range fails the same way on worker and interpreter.
 */
#ifndef CARAMEL_NORM_H
#define CARAMEL_NORM_H

#include <stdint.h>

#define CNRM_SQRT_FBITS 10          /* fractional bits kept on the sqrt */

/* round(a / b) for b > 0, half away from zero. */
static inline int64_t cnrm_rdiv(int64_t a, int64_t b)
{
    return (a >= 0) ? (a + b / 2) / b : -((-a + b / 2) / b);
}

/* floor(sqrt(n)) for n >= 0, exact, integer-only (bit-by-bit; no division). */
static inline uint64_t cnrm_isqrt(uint64_t n)
{
    uint64_t x = 0;
    uint64_t bit = (uint64_t)1 << 62;
    while (bit > n) { bit >>= 2; }
    while (bit) {
        if (n >= x + bit) { n -= x + bit; x = (x >> 1) + bit; }
        else              { x >>= 1; }
        bit >>= 2;
    }
    return x;
}

/* layernorm over a gathered axis-group: `len` register values `in` (scale
 * M = 10^q) -> `len` normalized register values `out` (scale M). Two-pass
 * (mean via S, variance via Σc^2), then out_i = c_i / sqrt(var) re-encoded.
 * Bit-exact: identical integer arithmetic + exact isqrt on both sides. */
static inline void cnrm_layernorm_group(const int32_t *in, int32_t *out,
                                        int32_t len, int64_t M)
{
    int32_t i;
    int64_t S = 0;
    int64_t SS = 0;
    uint64_t R;
    if (len <= 0) { return; }
    for (i = 0; i < len; i++) { S += in[i]; }
    for (i = 0; i < len; i++) {
        int64_t c = (int64_t)len * in[i] - S;   /* = len*(x_i - mean) */
        SS += c * c;
    }
    /* R ~= sqrt(len*SS) << CNRM_SQRT_FBITS  (len*SS = len^3 * var * M^2). */
    R = cnrm_isqrt(((uint64_t)((int64_t)len * SS)) << (2 * CNRM_SQRT_FBITS));
    for (i = 0; i < len; i++) {
        int64_t c = (int64_t)len * in[i] - S;
        if (R == 0) { out[i] = 0; continue; }   /* zero variance -> all zero */
        /* out_i = round( c_i * M * len / sqrt(len*SS) ), fractional-scaled.
         * Multiply (not shift) so a negative numerator stays well-defined. */
        int64_t num = c * M * (int64_t)len * ((int64_t)1 << CNRM_SQRT_FBITS);
        out[i] = (int32_t)cnrm_rdiv(num, (int64_t)R);
    }
}

#endif /* CARAMEL_NORM_H */
