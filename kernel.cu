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

#include "X_32.h"
#include "X_512.h"
#include "X_1024.h"

#define MAX_N 2048

using namespace std;

constexpr auto MAX_NUMBER_THREADS = 1024;

cudaError_t solveMatrixWithCuda(int numOfThreads, int dimension);
void PrintMatrix(double ar[][MAX_N], int n, int m, bool isInverted);
void InverseOfMatrix(double matrix[][MAX_N], int dimension);

__global__ void multiplyKernel(double* MatrixA, double* vecB, double* VecSol, int dimension, int numOfThreads)
{
	for (int i = 0; i <= dimension/numOfThreads; i++) {
		int j = (threadIdx.x + numOfThreads * i) + (blockIdx.x * numOfThreads);
		int k = dimension * j;

		for (int z = 0; z < dimension; z++) {
			VecSol[j] += MatrixA[k + z] * vecB[z];
		}
	}
}

// Function to print the matrix. 
void PrintMatrix(double ar[][MAX_N], int n, int m, bool isInverted)
{
	for (int i = 0; i < n; i++) {
		if (isInverted) {
			for (int j = n; j < m; j++) {
				printf("%.3f ", ar[i][j]);
			}
		}
		else {
			for (int j = 0; j < m; j++) {
				printf("%.3f ", ar[i][j]);
			}
		}
		printf("\n");
	}
	return;
}

// Function to perform the inverse operation on a matrix
void InverseOfMatrix(double matrix[][MAX_N], int dimension)
{
	double temp;

	for (int i = 0; i < dimension; i++) {
		// Create an identity matrix next to the matrix we are inverting
		for (int j = 0; j < 2 * dimension; j++) {
			if (j == (i + dimension))
				matrix[i][j] = 1;
		}
	}

	// Interchange/swap the rows
	for (int i = dimension - 1; i > 0; i--) {
		 if (matrix[i - 1][0] < matrix[i][0]) 
		 for (int j = 0; j < 2 * dimension; j++) {  
			 temp = matrix[i][j]; 
			 matrix[i][j] = matrix[i - 1][j]; 
			 matrix[i - 1][j] = temp; 
		 }
	}

	// Replace a row by sum of itself and a constant multiple of another row of the matrix 
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			if (j != i) {
				temp = matrix[j][i] / matrix[i][i];
				for (int k = 0; k < 2 * dimension; k++) {
					matrix[j][k] -= matrix[i][k] * temp;
				}
			}
		}
	}

	// Multiply each row by a nonzero integer and divide row element by the diagonal element 
	for (int i = 0; i < dimension; i++) {
		temp = matrix[i][i];
		for (int j = 0; j < 2 * dimension; j++) {
			matrix[i][j] = matrix[i][j] / temp;
		}
	}

	return;
}


int main(int argc, char* argv[])
{
	// Getting values for dimension and number of threads
	int dimension;
	int numOfThreads;

	if (argc != 3 || argv[1] == NULL || argv[2] == NULL ||
	argv[1] == "-h" || argv[1] == "--help" || argv[1] == "--h") {
		cout << "ParallelMatrixSolver.exe <Dimension (n) of Matrix = 10, 32, 512, or 1024> <# threads>" << endl;
		return 0;
	}
	else {
		if (argv[1] != NULL) {
			dimension = stoi(argv[1]);
			if (!(dimension == 10 || dimension == 32 || dimension == 512 || dimension == 1024)) {
				cout << "Dimension must be 10, 32, 512, or 1024" << endl;
				return -1;
			}
		}
		if (argv[2] != NULL) {
			numOfThreads = stoi(argv[2]);
		}
	}

	cout << "\nDimension of A = " << dimension;
	cout << "\nNumber of Threads = " << numOfThreads << endl;
	cout << "Calculating..." << endl;

	solveMatrixWithCuda(numOfThreads, dimension);
	
	return 0;
}

cudaError_t solveMatrixWithCuda(int numOfThreads, int dimension) { 
	cudaError_t cudaStatus = cudaError_t::cudaErrorDeviceUninitilialized;
	GpuTimer gpuTimer; // Struct for timing the GPU

	// Initialize sparse 2048 * 2048 matrices in CPU memory
	static double matrixA[MAX_N][MAX_N] = { 0 };
	static double solutionX[MAX_N][MAX_N] = { 0 };

	// Allocate from float matrices to the double matrix
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			if (dimension == 10) {
				matrixA[i][j] = A_10[i][j];
			}
			if (dimension == 32) {
				matrixA[i][j] = A_32[i][j];
			}
			if (dimension == 512) {
				matrixA[i][j] = A_512[i][j];
			}
			if (dimension == 1024) {
				matrixA[i][j] = A_1024[i][j];
			}
		}
	}

	// As per assignment instructions, if the matrix has dimension n >= 512 we don't invert and instead multiply 
	// the values directly with the solution matrix
	if (dimension == 10 || dimension == 32) {
		InverseOfMatrix(matrixA, dimension);
	}

	double *dev_inverseA, *dev_vectorB, *dev_solution = nullptr;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate memory for the input matrix, the vectorB, and then the solution
	cudaStatus = cudaMallocManaged((void**)& dev_inverseA, dimension * dimension * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for matrix failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_vectorB, dimension * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for vector b failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_solution, dimension * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for solution vector x failed!");
		goto Error;
	}

	// Copy over values from the input matrices to the CUDA memory
	// Note that the index which you copy from will vary depending on
	// whether or not you inverted the matrix
	for (int i = 0; i < dimension; i++) {
		int k = 0;
		if (dimension == 10 || dimension == 32) {
			for (int j = dimension; j < (2 * dimension); j++) {
				dev_inverseA[(i * dimension) + k] = matrixA[i][j];
				k++;
			}
		}
		if (dimension == 512 || dimension == 1024) {
			for (int j = 0; j < dimension; j++) {
				dev_inverseA[(i * dimension) + k] = matrixA[i][j];
				k++;
			}
		}
	}

	for (int i = 0; i < dimension; i++) {
		if (dimension == 10) {
			dev_vectorB[i] = b_10[i][0];
		}

		if (dimension == 32) {
			dev_vectorB[i] = b_32[i][0];
		}

		if (dimension == 512) {
			dev_vectorB[i] = X_512[i][0];
		}

		if (dimension == 1024) {
			dev_vectorB[i] = X_1024[i][0];
		}
	}

	int numBlocks = ((numOfThreads + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
	int threadsPerBlock = ((numOfThreads + (numBlocks - 1)) / numBlocks);

	/*************************************** Parrallel Part of Execution **********************************************/
	gpuTimer.Start();
	multiplyKernel << <numBlocks, threadsPerBlock >> > (dev_inverseA, dev_vectorB, dev_solution, dimension, threadsPerBlock);
	gpuTimer.Stop();
	/******************************************************************************************************************/
	float timeElapsed = gpuTimer.Elapsed();

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching one of the kernels!\n", cudaStatus);
		goto Error;
	}
	
	// Copy over values back from CUDA memory and print them
	for (int i = 0; i < dimension; i++) {
		solutionX[i][0] = dev_solution[i];
	}

	PrintMatrix(solutionX, dimension, 1, false);
	printf("-- Number of Threads: %d -- Dimension: %d -- Execution Time (ms): %g \n", numOfThreads, dimension, timeElapsed);

Error:
	cudaFree(dev_inverseA);
	cudaFree(dev_vectorB);
	cudaFree(dev_solution);

	return cudaStatus;
}
