/* This program can be used to transpose a matrix. Both the input and output
 * are in Matrix Market format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mmio.h"

#define N  512
#define M  512

#define REP 10


void transpose(int m, int n, float *A, float *B) {
   int i, j;

   for(i=0; i<m; i++) {
      for(j=0; j<n; j++) {
         B[i+j*m] = A[i*n+j];
      }
   }
}
