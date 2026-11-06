/* Checks the built library's CPU SGEMM (GPU disabled) against a naive triple
 * loop, to tell whether a GPU/CPU mismatch is the GPU's fault.
 * Usage: probe_cpu <n>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cblas.h"

int main(int argc, char **argv) {
  const int n = (argc > 1) ? atoi(argv[1]) : 64;
#ifdef _WIN32
  _putenv_s("OPENBLAS_RDNA2_GPU", "0");
#else
  setenv("OPENBLAS_RDNA2_GPU", "0", 1);
#endif
  float *A = malloc((size_t)n * n * sizeof(float));
  float *B = malloc((size_t)n * n * sizeof(float));
  float *C = calloc((size_t)n * n, sizeof(float));
  for (int i = 0; i < n * n; ++i) {
    A[i] = (float)((i % 7) - 3) / 2.0f;
    B[i] = (float)((i % 5) - 2) / 4.0f;
  }
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);

  double maxd = 0.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double acc = 0.0;
      for (int p = 0; p < n; ++p) acc += (double)A[i + p * n] * (double)B[p + j * n];
      double d = fabs((double)C[i + j * n] - acc);
      if (d > maxd) maxd = d;
    }
  printf("n=%d CPU cblas_sgemm vs naive: max|diff| = %.3e  %s\n", n, maxd,
         (maxd < 1e-3) ? "MATCH" : "MISMATCH");
  return 0;
}
