#include "mm_kernel.h"
#include <immintrin.h>
#define VSIZE (256 / 8 / sizeof(float))

void matrix_mult(int m, int n, int p, float *A, float *B, float *C) {
  int i, j, k;
  __m256 temp;
  __m256 temp_i;
  float rest;
  float res[VSIZE];
  #pragma omp parallel for private(i,j,k,temp,temp_i,rest,res)
  for(i=0; i<m; i++) {
    for(j=0; j<p; j++) {
      temp = _mm256_set1_ps(0.0);
      rest = 0;
      for(k=0; k < (n - n%VSIZE) ; k += VSIZE) {
        temp_i = _mm256_mul_ps(_mm256_loadu_ps(&A[i*n+k]), _mm256_loadu_ps(&B[j*n+k]));
        temp = _mm256_add_ps(temp, temp_i);
      }
      for(; k < n; ++k){
        rest += A[i*n+k]*B[j*n+k];
      }
      // Finish vectorizing to save
      _mm256_storeu_ps(res, temp);
      for(unsigned int index = 0; index < VSIZE; ++index) // Surely will be unrolled
        rest += res[index];
      C[i*p+j] = rest;
    }
  }
}
