/* Test for the packed int8 GEMM extension (cblas_gemm_i8 / cblas_gemm_i8_scale).
 *
 * The reference here is computed directly from the packed buffers with the
 * documented op()/stride rules, so it does not share the packing code in
 * interface/gemm_i8.c: a wrong transpose, stride or scale shows up as a
 * mismatch rather than as agreement between two copies of the same mistake.
 *
 * Every case runs twice, once with the GPU path enabled and once with
 * OPENBLAS_RDNA2_GPU=0, and the two results must be identical. The int32 path
 * is exact (32-bit two's-complement accumulation), so the comparison against
 * the integer reference is exact too; the scaled path is compared with f32
 * tolerance because it converts to float.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cblas.h"

/* Relative tolerance for the float entry point: the CPU path scales an int64
 * accumulator while the device kernel accumulates in float, so the two cannot
 * be bit identical. */
#define I8_FLOAT_TOL 1e-5

#ifdef _WIN32
static void env_set(const char *k, const char *v) { _putenv_s(k, v); }
#else
static void env_set(const char *k, const char *v) { setenv(k, v, 1); }
#endif

static int failures = 0;
static int checks = 0;

#define CHECK(cond, what)                                                    \
  do {                                                                       \
    ++checks;                                                                \
    if (!(cond)) {                                                           \
      printf("FAIL: %s (line %d)\n", what, __LINE__);                        \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

static uint32_t rng_state = 0x9e3779b9u;

static uint32_t rng(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

/* Packed word from four int8 values, value 0 in the low byte. */
static int32_t mk_word(int b0, int b1, int b2, int b3) {
  const uint32_t u = ((uint32_t)(uint8_t)b0) | ((uint32_t)(uint8_t)b1 << 8) |
                     ((uint32_t)(uint8_t)b2 << 16) | ((uint32_t)(uint8_t)b3 << 24);
  int32_t v;
  memcpy(&v, &u, sizeof(v));
  return v;
}

/* Signed byte dot of two packed words, wrapped to 32 bits. */
static uint32_t dot4(uint32_t a, uint32_t b) {
  uint32_t acc = 0;
  int k;
  for (k = 0; k < 4; ++k) {
    const int x = (int)(int8_t)(uint8_t)((a >> (8 * k)) & 0xffu);
    const int y = (int)(int8_t)(uint8_t)((b >> (8 * k)) & 0xffu);
    acc += (uint32_t)(x * y);
  }
  return acc;
}

static int sum4(uint32_t w) {
  int s = 0, k;
  for (k = 0; k < 4; ++k) s += (int)(int8_t)(uint8_t)((w >> (8 * k)) & 0xffu);
  return s;
}

/* Fill `count` words of the packed storage. */
static void fill_words(int32_t *m, size_t count) {
  size_t i;
  for (i = 0; i < count; ++i) m[i] = (int32_t)rng();
}

typedef struct {
  int m, n, k4;
  int transa, transb;
  int lda, ldb, ldc;
  int ta, tb; /* 1 when transposed */
} shape;

/* Element accessors for the stored operands. ld is the row stride in a
 * row-major call and the row count in a column-major one, which is the usual
 * BLAS meaning and exactly what the entry point documents. */
static int32_t op_a(const shape *s, const int32_t *a, int i, int p, int rm) {
  if (rm) return s->ta ? a[p * s->lda + i] : a[i * s->lda + p];
  return s->ta ? a[p + i * s->lda] : a[i + p * s->lda];
}

static int32_t op_b(const shape *s, const int32_t *b, int p, int j, int rm) {
  if (rm) return s->tb ? b[j * s->ldb + p] : b[p * s->ldb + j];
  return s->tb ? b[j + p * s->ldb] : b[p + j * s->ldb];
}

/* Reference for cblas_gemm_i8, computed from the packed buffers. */
static void ref_i8(const shape *s, const int32_t *a, const int32_t *b,
                   int32_t *c, int order_rowmajor) {
  int i, j, p;

  for (j = 0; j < s->n; ++j) {
    for (i = 0; i < s->m; ++i) {
      uint32_t acc = 0;
      int32_t out;

      for (p = 0; p < s->k4; ++p) {
        const int32_t aw = op_a(s, a, i, p, order_rowmajor);
        const int32_t bw = op_b(s, b, p, j, order_rowmajor);
        acc += dot4((uint32_t)aw, (uint32_t)bw);
      }
      memcpy(&out, &acc, sizeof(out));
      if (order_rowmajor) c[i * s->ldc + j] = out;
      else                c[i + j * s->ldc] = out;
    }
  }
}

/* Reference for cblas_gemm_i8_scale, in double. */
static void ref_i8_scale(const shape *s, const int32_t *a, const int32_t *b,
                         float *c, int order_rowmajor, const float *sa,
                         const float *sb, int zp_b) {
  int i, j, p;

  for (j = 0; j < s->n; ++j) {
    int64_t rowsum = 0;
    if (zp_b != 0)
      for (p = 0; p < s->k4; ++p) {
        const int32_t bw = op_b(s, b, p, j, order_rowmajor);
        rowsum += sum4((uint32_t)bw);
      }

    for (i = 0; i < s->m; ++i) {
      uint32_t acc = 0;
      int32_t acc_s;
      int64_t corr;
      double v;

      for (p = 0; p < s->k4; ++p) {
        const int32_t aw = op_a(s, a, i, p, order_rowmajor);
        const int32_t bw = op_b(s, b, p, j, order_rowmajor);
        acc += dot4((uint32_t)aw, (uint32_t)bw);
      }
      memcpy(&acc_s, &acc, sizeof(acc_s));
      corr = (int64_t)acc_s - (int64_t)zp_b * rowsum;
      v = (double)corr;
      if (sa) v *= (double)sa[i];
      if (sb) v *= (double)sb[j];

      if (order_rowmajor) c[i * s->ldc + j] = (float)v;
      else                c[i + j * s->ldc] = (float)v;
    }
  }
}

static int run_one(const shape *s, int order_rowmajor, int scaled) {
  /* Leading dimensions are the usual BLAS ones for the requested order, so the
   * extent they multiply is the column count in a column-major call and the row
   * count in a row-major call. */
  const int aext = order_rowmajor ? (s->ta ? s->k4 : s->m) : (s->ta ? s->m : s->k4);
  const int bext = order_rowmajor ? (s->tb ? s->n : s->k4) : (s->tb ? s->k4 : s->n);
  const int crows = s->m, ccols = s->n;
  const size_t asz = (size_t)aext * s->lda;
  const size_t bsz = (size_t)bext * s->ldb;
  const size_t csz = (size_t)(order_rowmajor ? crows : ccols) * s->ldc + 8;

  int32_t *a = (int32_t *)malloc(asz * sizeof(int32_t));
  int32_t *b = (int32_t *)malloc(bsz * sizeof(int32_t));
  int32_t *ref = (int32_t *)malloc(csz * sizeof(int32_t));
  int32_t *gotc = (int32_t *)malloc(csz * sizeof(int32_t));
  int32_t *gotg = (int32_t *)malloc(csz * sizeof(int32_t));
  float *reff = (float *)malloc(csz * sizeof(float));
  float *gotfc = (float *)malloc(csz * sizeof(float));
  float *gotfg = (float *)malloc(csz * sizeof(float));
  float sa[256], sb[256];
  char label[192];
  int i, ok = 1;

  if (!a || !b || !ref || !gotc || !gotg || !reff || !gotfc || !gotfg) {
    printf("FAIL: allocation\n");
    ++failures;
    return 0;
  }

  fill_words(a, asz);
  fill_words(b, bsz);
  for (i = 0; i < 256; ++i) {
    sa[i] = 0.5f + 0.25f * (float)i;
    sb[i] = 1.0f - 0.125f * (float)i;
  }

  env_set("OPENBLAS_RDNA2_GPU", "0");
  if (scaled) {
    ref_i8_scale(s, a, b, reff, order_rowmajor, sa, sb, 7);
    for (i = 0; i < (int)csz; ++i) gotfc[i] = -12345.0f;
    if (cblas_gemm_i8_scale(order_rowmajor ? CblasRowMajor : CblasColMajor,
                            s->ta ? CblasTrans : CblasNoTrans,
                            s->tb ? CblasTrans : CblasNoTrans,
                            s->m, s->n, s->k4, a, s->lda, b, s->ldb, gotfc,
                            s->ldc, sa, sb, 7) != 0)
      ok = 0;
  } else {
    ref_i8(s, a, b, ref, order_rowmajor);
    for (i = 0; i < (int)csz; ++i) gotc[i] = (int32_t)0x7eadbeef;
    if (cblas_gemm_i8(order_rowmajor ? CblasRowMajor : CblasColMajor,
                      s->ta ? CblasTrans : CblasNoTrans,
                      s->tb ? CblasTrans : CblasNoTrans,
                      s->m, s->n, s->k4, a, s->lda, b, s->ldb, gotc,
                      s->ldc) != 0)
      ok = 0;
  }

  env_set("OPENBLAS_RDNA2_GPU", "1");
  if (scaled) {
    for (i = 0; i < (int)csz; ++i) gotfg[i] = -12345.0f;
    if (cblas_gemm_i8_scale(order_rowmajor ? CblasRowMajor : CblasColMajor,
                            s->ta ? CblasTrans : CblasNoTrans,
                            s->tb ? CblasTrans : CblasNoTrans,
                            s->m, s->n, s->k4, a, s->lda, b, s->ldb, gotfg,
                            s->ldc, sa, sb, 7) != 0)
      ok = 0;
  } else {
    for (i = 0; i < (int)csz; ++i) gotg[i] = (int32_t)0x7eadbeef;
    if (cblas_gemm_i8(order_rowmajor ? CblasRowMajor : CblasColMajor,
                      s->ta ? CblasTrans : CblasNoTrans,
                      s->tb ? CblasTrans : CblasNoTrans,
                      s->m, s->n, s->k4, a, s->lda, b, s->ldb, gotg,
                      s->ldc) != 0)
      ok = 0;
  }

  snprintf(label, sizeof(label),
           "%s %dx%dxK4=%d %s/%s lda=%d ldb=%d ldc=%d", scaled ? "scale" : "i8",
           s->m, s->n, s->k4, s->ta ? "T" : "N", s->tb ? "T" : "N", s->lda,
           s->ldb, s->ldc);

  /* Result elements, plus the untouched padding when ldc has slack. The int
   * path must agree bit for bit; the float path accumulates in a different
   * precision on each side, so it is compared with a relative tolerance. */
  {
    double worst = 0.0;
    int worst_idx = -1;

    for (i = 0; i < (int)csz; ++i) {
      if (scaled) {
        const double d = fabs((double)gotfc[i] - (double)gotfg[i]) /
                         (fabs((double)gotfc[i]) + 1.0);

        if (d > worst) { worst = d; worst_idx = i; }
      } else if (gotc[i] != gotg[i]) {
        worst = 2.0;
        worst_idx = i;
        break;
      }
    }

    ok = scaled ? (worst <= I8_FLOAT_TOL) : (worst < 1.0);
    if (!ok)
      printf("  %s: GPU and CPU differ at idx %d (rel %.3g)\n", label,
             worst_idx, worst);
  }
  CHECK(ok, label);

  /* Exact check for int32, tolerance check for the float path. */
  ok = 1;
  for (i = 0; i < ccols; ++i) {
    int r;
    for (r = 0; r < crows; ++r) {
      const int idx = order_rowmajor ? (r * s->ldc + i) : (r + i * s->ldc);
      if (scaled) {
        const double d = fabs((double)gotfg[idx] - (double)reff[idx]);
        const double scale = fabs((double)reff[idx]) + 1.0;

        if (d > 1e-4 * scale) {
          printf("    idx %d (row %d col %d): got %.9g ref %.9g\n", idx, r, i,
                 (double)gotfg[idx], (double)reff[idx]);
          ok = 0;
          break;
        }
      } else if (gotg[idx] != ref[idx]) {
        ok = 0;
        break;
      }
    }
    if (!ok) break;
  }
  if (!ok) printf("  %s: mismatch against reference at column %d\n", label, i);
  CHECK(ok, label);

  free(a); free(b); free(ref); free(gotc); free(gotg);
  free(reff); free(gotfc); free(gotfg);
  return ok ? 1 : 0;
}

int main(void) {
  const int ms[] = {1, 3, 17, 64, 131};
  const int ns[] = {1, 5, 9, 64, 37};
  const int ks[] = {1, 2, 7, 16, 33};
  int order, tr, t;
  clock_t t0;
  double gpu_ms, cpu_ms;

  printf("cblas_gemm_i8 / cblas_gemm_i8_scale\n");

  for (order = 0; order < 2; ++order) {
    for (tr = 0; tr < 4; ++tr) {
      for (t = 0; t < 5; ++t) {
        shape s;
        const int padded = (t & 1);

        s.m = ms[t];
        s.n = ns[t];
        s.k4 = ks[t];
        s.ta = tr & 1;
        s.tb = (tr >> 1) & 1;
        s.transa = s.ta;
        s.transb = s.tb;
        if (order) { /* row-major: ld is the row stride */
          s.lda = (s.ta ? s.m : s.k4) + (padded ? 3 : 0);
          s.ldb = (s.tb ? s.k4 : s.n) + (padded ? 2 : 0);
          s.ldc = s.n + (padded ? 5 : 0);
        } else { /* column-major: ld is the number of rows */
          s.lda = (s.ta ? s.k4 : s.m) + (padded ? 3 : 0);
          s.ldb = (s.tb ? s.n : s.k4) + (padded ? 2 : 0);
          s.ldc = s.m + (padded ? 5 : 0);
        }
        run_one(&s, order, 0);
        run_one(&s, order, 1);
      }
    }
  }

  /* Accumulator overflow: forces sums past 2^31 so a 32-bit wrap is required
   * to match, exactly like v_dot4_i32_i8. */
  {
    shape s;
    int32_t *a, *b, *c;
    const int m = 3, n = 2;
    const int k4 = 40000;
    int j, i2, p;

    s.m = m; s.n = n; s.k4 = k4;
    s.ta = s.tb = 0; s.transa = s.transb = 0;
    s.lda = m; s.ldb = k4; s.ldc = m;

    a = (int32_t *)malloc((size_t)m * k4 * sizeof(int32_t));
    b = (int32_t *)malloc((size_t)n * k4 * sizeof(int32_t));
    c = (int32_t *)malloc((size_t)m * n * sizeof(int32_t));
    if (a && b && c) {
      for (i2 = 0; i2 < m * k4; ++i2) a[i2] = mk_word(-128, 127, -128, 127);
      for (i2 = 0; i2 < n * k4; ++i2) b[i2] = mk_word(127, -128, 127, -128);

      env_set("OPENBLAS_RDNA2_GPU", "1");
      CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k4, a,
                          m, b, k4, c, m) == 0,
            "wraparound call ok");

      /* Independent 64-bit total, then reduced modulo 2^32. */
      for (j = 0; j < n; ++j)
        for (i2 = 0; i2 < m; ++i2) {
          int64_t total = 0;
          int32_t want;
          for (p = 0; p < k4; ++p) {
            const int32_t aw = a[i2 + p * m];
            const int32_t bw = b[p + j * k4];
            int q;
            for (q = 0; q < 4; ++q) {
              const int x = (int)(int8_t)(uint8_t)(((uint32_t)aw >> (8 * q)) & 0xffu);
              const int y = (int)(int8_t)(uint8_t)(((uint32_t)bw >> (8 * q)) & 0xffu);
              total += (int64_t)x * (int64_t)y;
            }
          }
          want = (int32_t)(uint32_t)(uint64_t)total;
          CHECK(c[i2 + j * m] == want, "wraparound matches 64-bit reference");
          (void)want;
        }
    } else {
      ++failures;
    }
    free(a); free(b); free(c);
  }

  /* k4 == 0 zeroes C, and the remaining arguments are still validated. */
  {
    int c[6];
    int32_t a[1] = {0}, b[1] = {0};
    int i2;

    for (i2 = 0; i2 < 6; ++i2) c[i2] = 99;
    CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 3, 0, a, 2,
                        b, 1, c, 2) == 0,
          "k4=0 accepted");
    for (i2 = 0; i2 < 6; ++i2) CHECK(c[i2] == 0, "k4=0 zeroes C");
  }

  /* Argument validation. */
  {
    int32_t a[8] = {0}, b[8] = {0};
    int c[8] = {0};

    CHECK(cblas_gemm_i8((enum CBLAS_ORDER)999, CblasNoTrans, CblasNoTrans, 2, 2,
                        2, a, 2, b, 2, c, 2) == -1,
          "bad order rejected");
    CHECK(cblas_gemm_i8(CblasColMajor, (enum CBLAS_TRANSPOSE)999, CblasNoTrans,
                        2, 2, 2, a, 2, b, 2, c, 2) == -1,
          "bad transpose rejected");
    CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, -1, 2, 2, a, 2,
                        b, 2, c, 2) == -1,
          "negative m rejected");
    CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, a, 1,
                        b, 2, c, 2) == -1,
          "short lda rejected");
    CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, a, 2,
                        b, 2, c, 1) == -1,
          "short ldc rejected");
    CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, NULL,
                        2, b, 2, c, 2) == -1,
          "null A rejected");
    CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, a, 2,
                        b, 2, NULL, 2) == -1,
          "null C rejected");
    CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, 0, 2, 2, NULL,
                        2, NULL, 2, NULL, 2) == 0,
          "m=0 quick return");
    /* Conjugate codes are accepted: they are the identity on integers. */
    CHECK(cblas_gemm_i8(CblasColMajor, CblasConjTrans, CblasConjNoTrans, 2, 2, 2,
                        a, 2, b, 2, c, 2) == 0,
          "conjugate codes accepted");
    CHECK(cblas_gemm_i8_scale(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2,
                              a, 2, b, 2, (float *)c, 2, NULL, NULL, 0) == 0,
          "scale with NULL scales accepted");
  }

  /* Throughput at a size where the kernels are meant to be used. */
  {
    const int n = 512;
    const int k4 = 128;
    int32_t *a = (int32_t *)malloc((size_t)n * k4 * sizeof(int32_t));
    int32_t *b = (int32_t *)malloc((size_t)n * k4 * sizeof(int32_t));
    int32_t *c = (int32_t *)malloc((size_t)n * n * sizeof(int32_t));
    int pass;

    if (a && b && c) {
      fill_words(a, (size_t)n * k4);
      fill_words(b, (size_t)n * k4);

      /* Column-major NoTrans: A is n x k4 with lda = n, B is k4 x n with
       * ldb = k4, C is n x n with ldc = n. */
      env_set("OPENBLAS_RDNA2_GPU", "0");
      t0 = clock();
      CHECK(cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, k4, a,
                          n, b, k4, c, n) == 0,
            "throughput case accepted");
      cpu_ms = (double)(clock() - t0) * 1000.0 / (double)CLOCKS_PER_SEC;

      env_set("OPENBLAS_RDNA2_GPU", "1");
      t0 = clock();
      cblas_gemm_i8(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, k4, a, n, b,
                    k4, c, n);
      gpu_ms = (double)(clock() - t0) * 1000.0 / (double)CLOCKS_PER_SEC;

      printf("\n%dx%dxK4=%d (int8): gpu-path %.1f ms, cpu-path %.1f ms\n", n, n,
             k4, gpu_ms, cpu_ms);
      printf("(both paths are exact; a device-less or mismatched-arch build "
             "runs the CPU path in both)\n");
      (void)pass;
    }
    free(a); free(b); free(c);
  }

  /* The "FAILED" spelling is what test/Makefile's CHECK_TEST_SUMMARY greps
   * for. */
  printf("\n%s (%d checks, %d failures)\n",
         failures ? "INT8 API TEST FAILED" : "ALL INT8 API CASES MATCH", checks,
         failures);
  return failures ? 1 : 0;
}
