#include "mm_kernel.h"


void matrix_mult(int m, int n, int p, float *A, float *B, float *C) {
  int i, j, k;

  for(i=0; i<m; i++) {
    for(j=0; j<p; j++) {
      C[i*p+j]=0;
      for(k=0; k<n; k++) {
        C[i*p+j] += A[i*n+k]*B[k*p+j];
      }
    }
  }
}