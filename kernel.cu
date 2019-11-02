#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "A_10.h"
#include "b_10.h"
#include "A_32.h"
#include "b_32.h"
#include "A_512.h"
#include "b_512.h"
#include "A_1024.h"
#include "b_1024.h"

#define MAX_N 2048
#define N 512

using namespace std;

constexpr auto MAX_NUMBER_THREADS = 1024;

cudaError_t solveMatrixWithCuda(int numOfThreads);
void PrintMatrix(double ar[][MAX_N], int n, int m);
void PrintInverse(double ar[][MAX_N], int n, int m);
void InverseOfMatrix(double matrix[][MAX_N], int order);

__global__ void solveMatrixKernel(double* inverseA, double* vecB, double* VecSolx, int dimension, int numOfThreads)
{
	for (int i = 0; i < dimension/numOfThreads; i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = dimension * j;

		for (int z = 0; z < N; z++) {
			VecSolx[j] += inverseA[k + z] * vecB[z];
		}
		printf("%.3f \n", VecSolx[j]);
	}
}

// Function to Print matrix. 
void PrintMatrix(double ar[][MAX_N], int n, int m)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			cout << ar[i][j] << " ";
		}
		printf("\n");
	}
	return;
}

// Function to Print inverse matrix 
void PrintInverse(double ar[][MAX_N], int n, int m)
{
	for (int i = 0; i < n; i++) {
		for (int j = n; j < m; j++) {
			printf("%.3f ", ar[i][j]);
		}
		printf("\n");
	}
	return;
}

// Function to perform the inverse operation on the matrix. 
void InverseOfMatrix(double matrix[][MAX_N], int order)
{
	// Matrix Declaration. 

	double temp;

	// PrintMatrix function to print the element 
	// of the matrix. 
	//printf("=== Matrix ===\n");
	//PrintMatrix(matrix, order, order);

	// Create the augmented matrix 
	// Add the identity matrix 
	// of order at the end of original matrix. 
	for (int i = 0; i < order; i++) {

		for (int j = 0; j < 2 * order; j++) {

			// Add '1' at the diagonal places of 
			// the matrix to create a identity matirx 
			if (j == (i + order))
				matrix[i][j] = 1;
		}
	}

	// Interchange the row of matrix, 
	// interchanging of row will start from the last row 
	for (int i = order - 1; i > 0; i--) {

		// Swapping each and every element of the two rows 
		 if (matrix[i - 1][0] < matrix[i][0]) 
		 for (int j = 0; j < 2 * order; j++) { 
		 
			 // Swapping of the row, if above 
			 // condition satisfied. 
			 temp = matrix[i][j]; 
			 matrix[i][j] = matrix[i - 1][j]; 
			 matrix[i - 1][j] = temp; 
		 } 

		// Directly swapping the rows using pointers saves time 

		/*if (matrix[i - 1][0] < matrix[i][0]) {
			float* temp = matrix[i];
			matrix[i] = matrix[i - 1];
			matrix[i - 1] = temp;
		}*/
	}

	// Print matrix after interchange operations. 
	//printf("\n=== Augmented Matrix ===\n");
	//PrintMatrix(matrix, order, order * 2);

	// Replace a row by sum of itself and a 
	// constant multiple of another row of the matrix 
	for (int i = 0; i < order; i++) {

		for (int j = 0; j < order; j++) {

			if (j != i) {

				temp = matrix[j][i] / matrix[i][i];
				for (int k = 0; k < 2 * order; k++) {

					matrix[j][k] -= matrix[i][k] * temp;
				}
			}
		}
	}

	// Multiply each row by a nonzero integer. 
	// Divide row element by the diagonal element 
	for (int i = 0; i < order; i++) {

		temp = matrix[i][i];
		for (int j = 0; j < 2 * order; j++) {

			matrix[i][j] = matrix[i][j] / temp;
		}
	}

	// print the resultant Inverse matrix. 
	//printf("\n=== Inverse Matrix ===\n");
	//PrintInverse(matrix, order, 2 * order);

	return;
}


int main(int argc, char* argv[])
{
	int numOfThreads = 10;

	//cout << "\nDimension of A = " << N;
	cout << "\nNumber of Threads = \n" << numOfThreads;

	solveMatrixWithCuda(numOfThreads);
	
	return 0;

	//if (argc != 5 || argv[1] == NULL || argv[2] == NULL || argv[3] == NULL || argv[4] == NULL ||
	//	argv[1] == "-h" || argv[1] == "--help" || argv[1] == "--h") {
	//	cout << "ParallelMatrixSolver.exe <Matrix A> <Vector B> <# threads>" << endl;
	//	return 0;
	//}
	//else {
	//	if (argv[2] != NULL) {
	//		inputImgName = argv[2];
	//	}
	//	if (argv[3] != NULL) {
	//		outImgName = argv[3];
	//	}
	//	if (argv[4] != NULL) {
	//		numOfThreads = stoi(argv[4]);
	//	}
	//}

	//if (argv[1] != NULL && !strcmp(argv[1], "pool")) {
	//	cout << "Pooling" << endl;
	//	cudaError_t status = imagePoolingWithCuda(numOfThreads, inputImgName, outImgName);
	//}

	//return 0;
}

cudaError_t solveMatrixWithCuda(int numOfThreads) {
	cudaError_t cudaStatus = cudaError_t::cudaErrorDeviceUninitilialized;
	GpuTimer gpuTimer; // Struct for timing the GPU

	static double matrixA[MAX_N][MAX_N] = { 0 };

	//from float to double
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrixA[i][j] = A_512[i][j];
		}
	}

	InverseOfMatrix(matrixA, N);
	
	double *dev_inverseA, *dev_vectorB, *dev_solution = nullptr;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate memory for the input matrix, then it's adjoint, then it's inverse
	cudaStatus = cudaMallocManaged((void**)& dev_inverseA, N * N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_vectorB, N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_solution, N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy over values from the input matrices to the CUDA memory
	for (int i = 0; i < N; i++) {
		int k = 0;
		for (int j = N; j < (2*N); j++) {
			dev_inverseA[(i*N)+k] = matrixA[i][j];
			k++;
		}
	}

	for (int i = 0; i < N; i++) {
		dev_vectorB[i] = b_512[i][0];
	}

	int numBlocks = ((numOfThreads + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
	int threadsPerBlock = ((numOfThreads + (numBlocks - 1)) / numBlocks);
	/*************************************** Parrallel Part of Execution **********************************************/
	gpuTimer.Start();
	solveMatrixKernel << <numBlocks, threadsPerBlock >> > (dev_inverseA, dev_vectorB, dev_solution, N, threadsPerBlock);

	solveMatrixKernel << <numBlocks, threadsPerBlock >> > (dev_inverseA, dev_vectorB, dev_solution, N, threadsPerBlock);
	gpuTimer.Stop();
	/******************************************************************************************************************/
	printf("-- Number of Threads: %d -- Execution Time (ms): %g \n", numOfThreads, gpuTimer.Elapsed());

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "solveMatrixWithCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching one of the kernels!\n", cudaStatus);
		goto Error;
	}

	
	// Copy over values from the input matrices to the CUDA memory
	//for (int i = 0; i < N; i++) {
	//	vecX[i][0] = dev_solution[i];
	//}

	//PrintMatrix(vecX, N, 1);

Error:
	cudaFree(dev_inverseA);
	cudaFree(dev_vectorB);
	cudaFree(dev_solution);

	return cudaStatus;
}
