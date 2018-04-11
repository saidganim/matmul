#include "mm_kernel.h"


void matrix_mult(int m, int n, int p, float *A, float *B, float *C) {
  int i, j, k;

  #pragma omp parallel for
  for(i=0; i<m; i++) {
    for(j=0; j<p; j++) {
      float temp = 0;
      for(k=0; k<n; k++) {
        temp += A[i*n+k]*B[k*p+j];
      }
        C[i*p+j] = temp;
    }
  }
}
