// vec_add.cu: Parallel vector add using CUDA

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>


// Kernel function, runs on GPU
__global__ void add_vectors(float *a, float *b, float *c) {
		int i = blockIdx.x;
		
		c[i] = a[i] + b[i];
}


int main(void) {
		int count, i;
		
		// Find number of GPUs
		cudaGetDeviceCount(&count);
		printf("There are %d GPU devices in your system\n", count);

		int N = 10;	// Vector length

    // Create vectors a, b and c in the host (CPU)
		float *a = (float *)malloc(N*sizeof(float));
		float *b = (float *)malloc(N*sizeof(float));
		float *c = (float *)malloc(N*sizeof(float));
    
		// Initialize a and b
		for (i=0; i<N; i++) {
			a[i] = i - 0.5;
			b[i] = i*i - 3;
		}

		// Create a_dev, b_dev, c_dev on GPU
		float *a_dev, *b_dev, *c_dev;
		cudaMalloc((void **)&a_dev, N*sizeof(float));
		cudaMalloc((void **)&b_dev, N*sizeof(float));
		cudaMalloc((void **)&c_dev, N*sizeof(float));
		
    // Copy a, b and c vectors from host to GPU
		cudaMemcpy(a_dev, a, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_dev, b, N*sizeof(float), cudaMemcpyHostToDevice);
		
    // Parallel add c_dev[i] = a_dev[i] + b_dev[i]
		add_vectors<<< N, 1 >>>(a_dev, b_dev, c_dev);

		// Copy result from GPU to host (CPU)
		cudaMemcpy(c, c_dev, N*sizeof(float), cudaMemcpyDeviceToHost);
		
    // Free memory
		cudaFree(a_dev);
		cudaFree(b_dev);
		cudaFree(c_dev);

		// Print result on host (CPU)
		printf("\nVector Addition Result:\n");
		for (i=0; i<N; i++) {
			printf("a[%d] : %0.2f \t+\t", i, a[i]);
			printf("b[%d] : %0.2f \t=\t", i, b[i]);
			printf("c[%d] : %0.2f\n", i, c[i]);
		}
		
		return 0;
}
