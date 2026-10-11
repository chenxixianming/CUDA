#include<chrono>
#include<cmath>
#include<cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include<iostream>
#include<random>
#include<vector>
#include <cuda_runtime.h>
#include<cuda/cmath>
#include<cstdio>
#include<utility>

void CUDA_CHECK(cudaError_t err)
{
    if(err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

std::pair<std::mt19937, std::uniform_real_distribution<float>> random_generator(int seed = 123)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    return {generator, distribution};
}

bool save_json(const char *filename, const float *input, const float *kernel, const float *output, int H, int W, int K, double duration)
{
    std::ofstream file(filename);

    if(!file.is_open())
    {
        std::cerr << "Fialed to open: " << filename << "\n";
        return false;
    }

    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    file << "{\n";
    file << "   \"H\": " << H << ",\n";
    file << "   \"W\": " << W << ",\n";
    file << "   \"K\": " << K << ",\n";

    // file << "   \"input\": [\n";
    // for(int i = 0; i < H; i++)
    // {
    //     file << "       [";
    //     for(int j = 0; j < W; j++)
    //     {
    //         if(j > 0)
    //         {
    //             file << ", ";
    //         }
    //         file << input[i * W + j];
    //     }
    //     file << "]";
    //     if(i < H - 1)
    //     {
    //         file << ",";
    //     }
    //     file << "\n";
    // }
    // file << "   ],\n";

    // file << "   \"kernel\": [\n";
    // for(int i = 0; i < K; i++)
    // {
    //     file << "       [";
    //     for(int j = 0; j < K; j++)
    //     {
    //         if(j > 0)
    //         {
    //             file << ", ";
    //         }
    //         file << kernel[i * K + j];
    //     }
    //     file << "]";
    //     if(i < K - 1)
    //     {
    //         file << ",";
    //     }
    //     file << "\n";
    // }
    // file << "   ],\n";

    // file << "   \"output\": [\n";

    // for(int i = 0; i < H; i++)
    // {
    //     file << "       [";
    //     for(int j = 0; j < W; j++)
    //     {
    //         if(j > 0)
    //         {
    //             file << ", ";
    //         }
    //         file << output[i * W + j];
    //     }
    //     file << "]";
    //     if(i < H - 1)
    //     {
    //         file << ",";
    //     }
    //     file << "\n";
    // }
    // file << "   ],\n";
    file << "   \"duration\": " << duration << "\n";
    file << "}";

    file.close();

    return !file.fail();
}

void conv2d_cpu(const float *input, const float *kernel, 
    float *output, int H, int W, int K)
{
    const int R = K / 2;

    for(int i = 0; i < H; i++)
    {
        for(int j = 0; j < W; j++)
        {
            float sum = 0.0f;
            int idx = i * W + j;
            for(int k = - R; k <= R; k++)
            {
                for(int l = -R; l <= R; l++)
                {
                    int y = i + k;
                    int x = j + l;
                    if(-1 < y && y < H && -1 < x && x < W)
                    {
                        sum += input[y * W + x] * kernel[(k + R) * K + l + R];
                    }
                }
            }
            output[idx] = sum;
        }
    }
}

double cpu(const float *input, const float *kernel, float *output, const int H, const int W, const int K)
{
    const auto start_time = std::chrono::steady_clock::now();

    conv2d_cpu(input, kernel, output, H, W, K);

    const auto end_time = std::chrono::steady_clock::now();
    const double duration = std::chrono::duration<double, std::milli> (end_time - start_time).count();

    if (save_json("cpu.json", input, kernel, output, H, W, K, duration)) 
    {
        std::cout << "Results saved to: ./cpu.json\n";
    } 
    else 
    {
        std::cerr << "Failed to save cpu.json\n";
    }

    return duration;
}

__global__ void conv2d_gpu(const float *input, const float *kernel, 
    float *output, int H, int W, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row >= H || col >= W)
        return;

    int R = K / 2;
    float sum = 0.0f;

    for(int i = -R; i <= R; i++) {
        for(int j = -R; j <= R; j++) {
            int row1 = row + i;
            int col1 = col + j;
            if(row1 > -1 && row1 < H && col1 > -1 && col1 < W) {
                sum += input[row1 * W + col1] * kernel[(i + R) * K + j + R];
            }
        }
    }

    output[row * W + col] = sum;
}

double gpu(const float * input, const float *kernel, float *output, const int H, const int W, const int K)
{
    const size_t size = (size_t) H * W * sizeof(float);
    const size_t kernel_size = (size_t) K * K * sizeof(float);
    float *dev_input;
    float *dev_kernel;
    float *dev_output;

    // 
    CUDA_CHECK(cudaMalloc(&dev_input, size));
    CUDA_CHECK(cudaMalloc(&dev_kernel, kernel_size));
    CUDA_CHECK(cudaMalloc(&dev_output, size));

    CUDA_CHECK(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_kernel, kernel, kernel_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    const int warmup = 5;
    const int repeat = 100;

    // 预热：不放进计时区间
    for (int n = 0; n < warmup; ++n) 
    {
        conv2d_gpu<<<grid, block>>>(dev_input, dev_kernel, dev_output, H, W, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < repeat; i++)
    {
        conv2d_gpu<<<grid, block>>> (dev_input, dev_kernel, dev_output, H, W, K);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float duration = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&duration, start, stop));
    duration = duration / repeat;

    CUDA_CHECK(cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost));

    if (save_json("gpu.json", input, kernel, output, H, W, K, duration)) 
    {
        std::cout << "Results saved to: ./gpu.json\n";
    } 
    else 
    {
        std::cerr << "Failed to save gpu.json\n";
    }

    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_kernel));
    CUDA_CHECK(cudaFree(dev_output));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return duration;
}

__global__ void conv2d_gpu_shared(const float *input, const float *kernel, 
    float *output, int H, int W, int K)
{
    // Inside a kernel
    // 每个 block 有一块自己的动态共享内存
    extern __shared__ float tile[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int origin_x = blockDim.x * blockIdx.x;
    int origin_y = blockDim.y * blockIdx.y;

    int col = tx + origin_x;
    int row = ty + origin_y;

    int R = K / 2;

    // tile 的尺寸
    int tile_w = blockDim.x + K - 1;
    int tile_h = blockDim.y + K - 1;

    // 确定每个线程写的坐标：从 tid 开始，每隔 thread_count 个坐标写（把任务平均分配给每一个线程）
    int tid = ty * blockDim.x + tx;
    int thread_count = blockDim.x * blockDim.y;

    for(int i = tid; i < tile_w * tile_h; i += thread_count)
    {
        int sy = i / tile_w;
        int sx = i % tile_w;

        int iy = origin_y + sy - R;
        int ix = origin_x + sx - R;

        if(iy > -1 && iy < H && ix > -1 && ix < W)
        {
            tile[i] = input[iy * W + ix];
        }
        else
        {
            tile[i] = 0.0f;
        }
    }

    // 等待所有线程写完tile
    __syncthreads();

    if(row < H && col < W)
    {
        float sum = 0.0f;

        for(int i = -R; i <= R; i++)
        {
            for(int j = -R; j <= R; j++)
            {
                int sy = ty + R + i;
                int sx = tx + R + j;

                sum += tile[sy * tile_w + sx] * kernel[(i + R) * K + j + R];
            }
        }

        output[row * W + col] = sum;
    }

    return;
}

double gpu_shared(const float * input, const float *kernel, float *output, const int H, const int W, const int K)
{
    const size_t size = (size_t) H * W * sizeof(float);
    const size_t kernel_size = (size_t) K * K * sizeof(float);
    float *dev_input;
    float *dev_kernel;
    float *dev_output;

    CUDA_CHECK(cudaMalloc(&dev_input, size));
    CUDA_CHECK(cudaMalloc(&dev_kernel, kernel_size));
    CUDA_CHECK(cudaMalloc(&dev_output, size));

    CUDA_CHECK(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_kernel, kernel, kernel_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    size_t shared_bytes = (size_t) (block.x + K - 1) * (block.y + K - 1) * sizeof(float);

    const int warmup = 5;
    const int repeat = 100;

    // 预热：不放进计时区间
    for (int n = 0; n < warmup; ++n) 
    {
        conv2d_gpu_shared<<<grid, block, shared_bytes>>>(dev_input, dev_kernel, dev_output, H, W, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < repeat; i++)
    {
        conv2d_gpu_shared<<<grid, block, shared_bytes>>> (dev_input, dev_kernel, dev_output, H, W, K);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float duration = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&duration, start, stop));
    duration = duration / repeat;

    CUDA_CHECK(cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost));

    if (save_json("gpu_shared.json", input, kernel, output, H, W, K, duration)) 
    {
        std::cout << "Results saved to: ./gpu_shared.json\n";
    } 
    else 
    {
        std::cerr << "Failed to save gpu_shared.json\n";
    }

    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_kernel));
    CUDA_CHECK(cudaFree(dev_output));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return duration;
}

bool match_output(const float* cpu, const float* gpu, const int size, const float atol = 1e-4f, const float rtol = 1e-4f)
{
    double max_error = 0.0;
    int failed = 0;

    for (int i = 0; i < size; ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) {
            ++failed;
            continue;
        }

        double error = std::abs(
            static_cast<double>(cpu[i]) - gpu[i]
        );
        double limit = atol
                     + rtol * std::abs(static_cast<double>(cpu[i]));

        if (error > max_error) {
            max_error = error;
        }

        if (error > limit) {
            ++failed;
        }
    }

    std::cout << "Max absolute error (finite pairs): "
              << max_error << '\n';
    std::cout << "Failed elements: " << failed << '\n';

    return failed == 0;
}

int main()
{
    int H = 4096;
    int W = 4096;
    int K = 31;

    if (H <= 0 || W <= 0 || K <= 0 || K % 2 == 0) 
    {
        std::cerr << "H and W must be positive; K must be positive and odd.\n";
        return EXIT_FAILURE;
    }

    int size = H * W;
    int kernel_size = K * K;

    float *input = (float *) malloc(size * sizeof(float));
    float *kernel = (float*) malloc(kernel_size * sizeof(float));

    auto randGen = random_generator();
    for(int i = 0; i < size; i++)
    {
        input[i] = randGen.second(randGen.first);
    }
    for(int i = 0; i < kernel_size; i++)
    {
        kernel[i] = randGen.second(randGen.first);
    }

    float *output_cpu = (float *) malloc(size * sizeof(float));
    double cpu_duration = cpu(input, kernel, output_cpu, H, W, K);

    float *output_gpu = (float *) malloc(size * sizeof(float));
    double gpu_duration = gpu(input, kernel, output_gpu, H, W, K);

    float *output_gpu_shared = (float *) malloc(size * sizeof(float));
    double gpu_shared_duration = gpu_shared(input, kernel, output_gpu_shared, H, W, K); 

    double gpu_speedup = cpu_duration / gpu_duration;
    bool gpu_match = match_output(output_cpu, output_gpu, size);

    printf("GPU result match: ");
    if(gpu_match)
        printf("true\n");
    else
        printf("false\n");
    printf("GPU speedup: %lfx\n", gpu_speedup);

    double share_speedup = gpu_duration / gpu_shared_duration;
    bool share_match = match_output(output_gpu, output_gpu_shared, size);
    printf("Shared result match: ");
    if(share_match)
        printf("true\n");
    else
        printf("false\n");

    printf("Share speedup (compared to gpu): %lfx\n", share_speedup);

    free(input);
    free(kernel);
    free(output_cpu);
    free(output_gpu);
    free(output_gpu_shared);
    input = nullptr;
    kernel = nullptr;
    output_cpu = nullptr;
    output_gpu = nullptr;
    output_gpu_shared = nullptr;

    return 0;
}