#include <stdio.h>
#include <hip/hip_runtime.h>
// #include <random>
#include <iostream>
using namespace std;

#define DIM1 1000
#define DIM2 1000
#define DIM3 1000

#define SCALAR_T float

#define BLOCKDIM1 DIM1
#define BLOCKDIM2 DIM3
#define BLOCKDIM3 1
#define THREADS_PER_BLOCK (BLOCKDIM1*BLOCKDIM2*BLOCKDIM3)

#define GRIDDIM1 1
#define GRIDDIM2 1
#define GRIDDIM3 1

#define RANDSIZE 10

template<typename T>
__global__ void matrixMul_rowMajor_kernel(T* A, T* B, T* C, size_t dim_1, size_t dim_2, size_t dim_3) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;          // row
    int j = blockDim.y * blockIdx.y + threadIdx.y;          // col
    // printf(" i = %d, j = %d\n", i, j);
    T sum = 0;
    for (int k = 0; k < dim_2; k++) {
        sum += A[i * dim_2 + k] * B[k * dim_3 + j];
    }
    C[i * dim_3 + j] = sum;
}

template<typename T>
__global__ void matrixMul_colMajor_kernel(T* A, T* B, T* C, size_t dim_1, size_t dim_2, size_t dim_3) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;          // row
    int j = blockDim.y * blockIdx.y + threadIdx.y;          // col
    // printf(" i = %d, j = %d\n", i, j);
    T sum = 0;
    for (int k = 0; k < dim_2; k++) {
        sum += A[i + k*dim_1] * B[k + j*dim_2];
    }
    C[i + j*dim_1] = sum;
}


template<typename T>
void matrixMul_rowMajor_host(T* A, T* B, T* C, size_t dim_1, size_t dim_2, size_t dim_3) {
    for (int i = 0; i < dim_1; i++) {
	    for (int j = 0; j < dim_3; j++) {
            T sum = 0;
	        for (int k = 0; k < dim_2; k++) {
		        sum += A[i*dim_2 + k] * B[k*dim_3 + j];
	        }
            C[i*dim_3 + j] = sum;
	    }
    }
};

template<typename T>
void matrixMul_colMajor_host(T* A, T* B, T* C, size_t dim_1, size_t dim_2, size_t dim_3) {
    for (int i = 0; i < dim_1; i++) {
	    for (int j = 0; j < dim_3; j++) {
            T sum = 0;
	        for (int k = 0; k < dim_2; k++) {
		        sum += A[i + k*dim_1] * B[k + j*dim_2];
	        }
            C[i + j*dim_1] = sum;
	    }
    }
};

template<typename T>
void printMatrix_rowMajor(T* mat, size_t dim_1, size_t dim_2) {
    for (int i = 0; i < dim_1; i++) {
        for (int j = 0; j < dim_2; j++) {
            cout << mat[i*dim_2+j] << ",";
        }
        cout << endl;
    }
}

template<typename T>
void printMatrix_colMajor(T* mat, size_t dim_1, size_t dim_2) {
    for (int i = 0; i < dim_1; i++) {
        for (int j = 0; j < dim_2; j++) {
            cout << mat[i+j*dim_1] << ",";
        }
        cout << endl;
    }
}

template<typename T>
void initMatrix(T* mat, size_t dim_1, size_t dim_2, int randSize) {
    for (int i = 0; i < dim_1; i++) {
        for (int j = 0; j < dim_2; j++) {
	    mat[i*dim_2+j] = randSize > 1 ? static_cast<SCALAR_T>(rand() % randSize) :
		                                static_cast<SCALAR_T>(randSize);
	    }
    }
}

template<typename T>
void compareMatrix(T* mat1, T* mat2, size_t dim_1, size_t dim_2) {
    cout << " ---------- Matrix Multiplication Result Comparison ---------- " << endl;
    int num_diff = 0;
    for (int i = 0; i < dim_1; i++) {
        for(int j = 0; j < dim_2; j++)
        {
            if (mat1[i*dim_2+j] != mat2[i*dim_2+j]) {
                // cout << "Result difference found at " << i << "," << j << ": " << "CPU: " << mat1[i*dim_2+j] << " GPU: " <<  mat2[i*dim_2+j] << endl;
                num_diff += 1;
            }
            else {
                // cout << "[" << i << ", " << j << "]: CPU: " << mat1[i*dim_2+j] << ", GPU: " << mat2[i*dim_2+j] << endl;
            }
        }
    }
    cout << "Number of result differences : " << num_diff << endl;
}

int main() {
    SCALAR_T* A = new SCALAR_T[DIM1*DIM2];
    SCALAR_T* B = new SCALAR_T[DIM2*DIM3];
    SCALAR_T* C = new SCALAR_T[DIM1*DIM3];
    SCALAR_T* C_GPU = new SCALAR_T[DIM1*DIM3];

    SCALAR_T* d_A, *d_B, *d_C;

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double, std::nano> elapsed;

    // Initialize matrix
    initMatrix(A, DIM1, DIM2, RANDSIZE);
    initMatrix(B, DIM2, DIM3, RANDSIZE);
    initMatrix(C, DIM1, DIM3, 0);

    // Matrix Multiplication from Host
    std::cout << " ---------- CPU ----------" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    matrixMul_rowMajor_host(A, B, C, DIM1, DIM2, DIM3);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CPU Elapsed time: " << elapsed.count() << " ns" << std::endl;
    std::cout << "Dummy print C[0]: " << C[0] << std::endl;

    /*
    // Print in row major
    cout << "Matrix A :" << endl;
    printMatrix_rowMajor(A, DIM1, DIM2);
    cout << "Matrix B :" << endl;
    printMatrix_rowMajor(B, DIM2, DIM3);
    cout << "Matrix C :" << endl;
    printMatrix_rowMajor(C, DIM1, DIM3);
    */

    /*
    // Matrix Multiplication from Host
    initMatrix(C, DIM1, DIM3, 0);
    start = std::chrono::high_resolution_clock::now();
    matrixMul_colMajor_host(A, B, C, DIM1, DIM2, DIM3);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CPU Elapsed time: " << elapsed.count() << " ns" << std::endl;
    std::cout << "Dummy print C[0]: " << C[0] << std::endl;

    // Print in column major
    cout << "Matrix A :" << endl;
    printMatrix_colMajor(A, DIM1, DIM2);
    cout << "Matrix B :" << endl;
    printMatrix_colMajor(B, DIM2, DIM3);
    cout << "Matrix C :" << endl;
    printMatrix_colMajor(C, DIM1, DIM3); 
    */
    std::cout << " ---------- GPU ----------" << std::endl;
    // Create Event Time Mark
    hipEvent_t startHip, stopHip;
    hipEventCreate(&startHip);
    hipEventCreate(&stopHip);

    // Malloc memory in device
    hipMalloc(&d_A, DIM1*DIM2*sizeof(SCALAR_T));
    hipMalloc(&d_B, DIM2*DIM3*sizeof(SCALAR_T));
    hipMalloc(&d_C, DIM1*DIM3*sizeof(SCALAR_T));

    // Copy Host data to Device Memory
    hipMemcpy(d_A, A, DIM1*DIM2*sizeof(SCALAR_T), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, DIM2*DIM3*sizeof(SCALAR_T), hipMemcpyHostToDevice);
    hipMemset(d_C, 0, DIM1*DIM3*sizeof(SCALAR_T));

    // Device 0 Information : 
    // Total Global Memory: 68702699520 bytes
    // Shared Memory per Block: 65536 bytes
    // Max Threads per Block: 1024
    // Max Threads Dimension: [1024, 1024, 1024]
    // Max Grid Size: [2147483647, 65536, 65536]

    dim3 blockDims(BLOCKDIM1, BLOCKDIM2, BLOCKDIM3);
    dim3 gridDims(GRIDDIM1, GRIDDIM2, GRIDDIM3);
    cout << "Block Dim : (" << blockDims.x << ", " << blockDims.y << ", " << blockDims.z << "), Total # of threads per block = " << blockDims.x*blockDims.y*blockDims.z<< endl;
    cout << "Grid Dim : (" << gridDims.x << ", " << gridDims.y << ", " << gridDims.z << "), Total # of blocks = " << gridDims.x*gridDims.y*gridDims.z<< endl;

    hipEventRecord(startHip, 0);
    hipLaunchKernelGGL(matrixMul_rowMajor_kernel, blockDims, gridDims, 0, 0, d_A, d_B, d_C, DIM1, DIM2, DIM3);
    // hipLaunchKernelGGL(matrixMul_colMajor_kernel, blockDims, gridDims, 0, 0, d_A, d_B, d_C, DIM1, DIM2, DIM3);
    hipEventRecord(stopHip, 0);
    hipEventSynchronize(stopHip);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, startHip, stopHip);
    std::cout << "GPU Elapsed time: " << milliseconds*1e6 << " ns" << std::endl;

    hipMemcpy(C_GPU, d_C, DIM1 * DIM3 * sizeof(SCALAR_T), hipMemcpyDeviceToHost);

    // Comapre result of CPU and GPU
    compareMatrix(C, C_GPU, DIM1, DIM3);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipEventDestroy(startHip);
    hipEventDestroy(stopHip);

	delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_GPU;
    return 0;
};
