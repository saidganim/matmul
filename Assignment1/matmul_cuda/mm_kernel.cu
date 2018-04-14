

#include "mm_kernel.h"

#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>

__global__ void matrix_kernel(int m, int n, int p, float* A, float* B, float* C){

	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned j = i % p;
	i = i / p;

	if(i >= m)
		return;

	for(int k=0; k<n; k++) {
	        C[i*p+j] += A[i*n+k]*B[k*p+j];
	}

}


void matrix_mult(int m, int n, int p, float *A, float *B, float *C) {
  //int i, j, k;
	struct timeval start, end;
	int threadBlock = 512;
	float *dA, *dB, *dC;
	cudaMalloc(&dA, m * n * sizeof(float));
	cudaMalloc(&dB, p * n * sizeof(float));
	cudaMalloc(&dC, m * p * sizeof(float));

	//if(cudaGetLastError != cudaSuccess){
	//	printf("CUDA_ERROR\n");
	//	exit(1);
	//}

	cudaMemcpy(dA, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, p*n*sizeof(float), cudaMemcpyHostToDevice);
	gettimeofday(&start, 0);
	matrix_kernel<<<m * p / threadBlock + 1, threadBlock>>>(m,n,p,dA,dB,dC);
	cudaDeviceSynchronize();
	gettimeofday(&end, 0);
	printf("time without memory copy = %f\n", end.tv_sec + end.tv_usec/1000000.0 - (start.tv_sec + start.tv_usec / 1000000.0));
	cudaMemcpy(C, dC, m*p*sizeof(float), cudaMemcpyDeviceToHost);

//  for(i=0; i<m; i++) {
  //  for(j=0; j<p; j++) {
    //  C[i*p+j]=0;
     // for(k=0; k<n; k++) {
       // C[i*p+j] += A[i*n+k]*B[k*p+j];
//      }
  //  }
//  }
}
