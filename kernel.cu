#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "A_10.h";
#include "b_10.h";

#define N 10

using namespace std;

constexpr auto MAX_NUMBER_THREADS = 1024;

cudaError_t solveMatrixWithCuda(float* matrixA, float* vectorB, int dimension, int numOfThreads);

//__global void solveMatrixKernel(float** inverseMatrix, float** vector, int dimension, int numOfThreads) 
//{
//	
//
//}

// Function to get cofactor of A[p][q] in temp[][]. n is current 
// dimension of A[][] 
void getCofactor(float matrix[N][N], float temp[N][N], int p, int q, int n)
{
	int i = 0, j = 0;

	// Looping for each element of the matrix 
	for (int row = 0; row < n; row++)
	{
		for (int col = 0; col < n; col++)
		{
			//  Copying into temporary matrix only those element 
			//  which are not in given row and column 
			if (row != p && col != q)
			{
				temp[i][j++] = matrix[row][col];

				// Row is filled, so increase row index and 
				// reset col index 
				if (j == n - 1)
				{
					j = 0;
					i++;
				}
			}
		}
	}
}

/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][]. */
int determinant(float matrix[N][N], int n)
{
	int D = 0; // Initialize result 

	//  Base case : if matrix contains single element 
	if (n == 1)
		return matrix[0][0];

	float temp[N][N]; // To store cofactors 

	int sign = 1;  // To store sign multiplier 

	 // Iterate for each element of first row 
	for (int f = 0; f < n; f++)
	{
		// Getting Cofactor of A[0][f] 
		getCofactor(matrix, temp, 0, f, n);
		D += sign * matrix[0][f] * determinant(temp, n - 1);

		// terms are to be added with alternate sign 
		sign = -sign;
	}

	return D;
}

// Function to get adjoint of A[N][N] in adj[N][N]. 
void adjoint(float matrix[N][N], float adj[N][N])
{
	if (N == 1)
	{
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][] 
	int sign = 1; 
	float temp[N][N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			// Get cofactor of A[i][j] 
			getCofactor(matrix, temp, i, j, N);

			// sign of adj[j][i] positive if sum of row 
			// and column indexes is even. 
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the transpose of the cofactor matrix 
			adj[j][i] = (sign) * (determinant(temp, N - 1));
		}
	}
}

// Function to calculate and store inverse, returns false if 
// matrix is singular 
bool inverse(float matrix[N][N], float inverse[N][N])
{
	// Find determinant of A[][] 
	int det = determinant(matrix, N);
	if (det == 0)
	{
		cout << "Singular matrix, can't find its inverse";
		return false;
	}

	// Find adjoint 
	float adj[N][N];
	adjoint(matrix, adj);

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			inverse[i][j] = adj[i][j] / float(det);
		}
	}

	return true;
}
 
void display(float matrix[N][N])
{
	for (int i = 0; i < N; i++) {
		cout << "{ ";
		for (int j = 0; j < N; j++) {
			cout << matrix[i][j] << ", ";
		}
		cout << " }" << endl;
	}
}


int main(int argc, char* argv[])
{
	int numOfThreads = 10;
	int dimension = 10;

	float adj[N][N] = { 0 };  // To store adjoint of A[][] 

	float inv[N][N] = { 0 }; // To store inverse of A[][] 
	
	//cudaError_t status = adjointWithCuda(numOfThreads, &InputMatrix);

	cout << "\nDimension of A = " << N;
	cout << "\nNumber of Threads = " << numOfThreads;

	cout << "\nMatrix A =\n";
	display(A_10);

	cout << "\nThe Adjoint of A=\n";
	adjoint(A_10, adj);
	display(adj);

	cout << "\nThe Inverse of A=\n";
	if (inverse(A_10, inv))
		display(inv);

	cout << "\nx = Inverse of A * b = ";

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

cudaError_t solveMatrixWithCuda(float* inverseA, float* vectorB, int dimension, int numOfThreads) {
	cudaError_t cudaStatus = cudaError_t::cudaErrorDeviceUninitilialized;
	GpuTimer gpuTimer; // Struct for timing the GPU

	float* dev_inverseA;
	float* dev_vectorB;
	float* dev_solution;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate memory for the input matrix, then it's adjoint, then it's inverse
	cudaStatus = cudaMallocManaged((void**)& dev_inverseA, dimension * dimension * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_vectorB, dimension * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocManaged((void**)& dev_solution, dimension * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy over values from the input matrices to the CUDA memory
	// TODO

	int numBlocks = ((numOfThreads + (MAX_NUMBER_THREADS - 1)) / MAX_NUMBER_THREADS);
	int threadsPerBlock = ((numOfThreads + (numBlocks - 1)) / numBlocks);

	/*************************************** Parrallel Part of Execution **********************************************/
	gpuTimer.Start();
	//solveMatrixKernel << <numBlocks, threadsPerBlock >> > (dev_inverseA, dev_vectorB, dev_solution, dimension, threadsPerBlock);
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

Error:
	cudaFree(dev_inverseA);
	cudaFree(dev_vectorB);
	cudaFree(dev_solution);
	return cudaStatus;
}
