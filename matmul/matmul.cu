#include<chrono>
#include<cmath>
#include<cstdlib>
#include<iostream>
#include<random>
#include<vector>
#include <cuda_runtime.h>
#include<cuda/cmath>

void matrix_mul_cpu(const float *A, const float *B, float *C, int N)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return;
}

double calculate_gflops(int N, double time)
{
    const double num_ops = 2.0 * N * N * N;
    return num_ops / (time * 1.0e6);
}

double cpu(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, int N)
{
    // save start time
    const auto start_time = std::chrono::steady_clock::now();

    // matrix mulplication by cpu
    matrix_mul_cpu(A.data(), B.data(), C.data(), N);
    
    // save end time
    const auto end_time = std::chrono::steady_clock::now();
    const double cpu_duration = std::chrono::duration<double, std::milli> (end_time - start_time).count();

    double gflops_cpu = calculate_gflops(N, cpu_duration);

    // make sure compiler do not eliminate dead code
    double count = 0.0;
    for(auto value : C)
        count += value;

    printf("CPU time: %lf ms\n", cpu_duration);
    printf("CPU gflops: %lf GFLOPS\n", gflops_cpu);
    printf("Check sum: %lf\n", count);

    return cpu_duration;
}

__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= N || col >= N)
        return;

    float sum = 0.0f;

    for(int i = 0; i < N; i++)
    {
        sum += A[row * N + i] * B[i * N + col];
    }

    C[row * N + col] = sum;
}

void CUDA_CHECK(cudaError_t err)
{
    if(err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

double gpu(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int N, size_t bytes)
{
    float *devA = nullptr, *devB = nullptr, *devC = nullptr;
    
    CUDA_CHECK(cudaMalloc(&devA, bytes)); 
    CUDA_CHECK(cudaMalloc(&devB, bytes));
    CUDA_CHECK(cudaMalloc(&devC, bytes));

    CUDA_CHECK(cudaMemcpy(devA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devB, B.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // save start time
    const auto start_time = std::chrono::steady_clock::now();
    
    matrix_mul_gpu<<<grid, block>>>(devA, devB, devC, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // save end time
    const auto end_time = std::chrono::steady_clock::now();
    const double gpu_duration = std::chrono::duration<double, std::milli> (end_time - start_time).count();

    double gflops_gpu = calculate_gflops(N, gpu_duration);

    CUDA_CHECK(cudaMemcpy(C.data(), devC, bytes, cudaMemcpyDeviceToHost));

    // make sure compiler do not eliminate dead code
    double count = 0.0;
    for(auto value : C)
        count += value;

    printf("GPU time: %lf ms\n", gpu_duration);
    printf("GPU gflops: %lf GFLOPS\n", gflops_gpu);
    printf("Check sum: %lf\n", count);

    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC));

    return gpu_duration;
}

int main()
{
    // 矩阵大小
    constexpr int N = 256;
    
    // 元素个数 & 占据字节数
    const size_t element_count = (size_t) N * N;
    const size_t byte_count = element_count * sizeof(float);

    printf("Matrix size: %d * %d\n", N, N);
    printf("Number of elements: %zu\n", element_count);
    printf("Memory: %zu KiB\n", byte_count / 1024);

    std::vector<float> A(element_count), B(element_count), C_cpu(element_count, 0.0f), C_gpu(element_count, 0.0f);

    // set random seed
    std::mt19937 generator(123);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

    // generate matrices
    for(size_t i = 0; i < element_count; i++)
    {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    double cpu_duration = cpu(A, B, C_cpu, N);
    double gpu_duration = gpu(A, B, C_gpu, N, byte_count);

    double speedup = cpu_duration / gpu_duration;
    printf("Speedup: %lfx", speedup);
    
    return 0;
}