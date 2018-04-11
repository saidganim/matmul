extern "C"
{
#include "mm_kernel.h"
}
#include <cuda.h>


__global__ void matrix_kernel(int m, int n, int p, float* A, float* B, float* C){

	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned j = i % p;
	i = i / p;
	
	if(i >= m || j >= p)
		return;
	
	for(int k=0; k<n; k++) {
	        C[i*p+j] += A[i*n+k]*B[k*p+j];
	}

}


extern "C"
void matrix_mult(int m, int n, int p, float *A, float *B, float *C) {
  //int i, j, k;
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

	matrix_kernel<<<m * n / threadBlock + 1, threadBlock>>>(m,n,p,dA,dB,dC);
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
