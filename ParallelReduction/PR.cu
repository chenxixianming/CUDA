#include<algorithm>
#include<chrono>
#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cuda/cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include<iostream>
#include <limits>
#include<random>
#include<utility>
#include<vector>

double sum_cpu(const float* input, int N)
{
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        sum += static_cast<double>(input[i]);
    }

    return sum;
}

__global__ void sum_one_block(const float* input, float* output, int N)
{
    __shared__ float shared[256];

    int tid = threadIdx.x;

    // TODO 1：
    // tid < N 时加载 input[tid]，否则写入 0
    // 每个线程都必须初始化自己的 shared[tid]
    if(tid < N)
    {
        shared[tid] = input[tid];
    }
    else
    {
        shared[tid] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride) {
            // TODO 2：将另一半的数据加到 shared[tid]
            shared[tid] += shared[tid + stride];
        }

        // TODO 3：保证本轮写入完成后再开始下一轮
        __syncthreads();
    }

    if (tid == 0) {
        // TODO 4：将最终结果写入 output[0]
        output[0] = shared[0];
    }
}

__global__ void sum_blocks(const float* input, float* partial, int N)
{
    __shared__ float shared[256];

    int tid = threadIdx.x;

    // TODO 1：计算当前线程对应的全局输入下标
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO 2：加载 input[idx]，越界时补零
    shared[tid] = idx < N ? input[idx] : 0;

    __syncthreads();

    // block 内归约：和你之前写的一样
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

__global__ void sum_blocks_stride(const float* input, float* partial, int N)
{
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int step = blockDim.x * gridDim.x;

    // 每个线程自己的局部累加值，通常保存在寄存器中
    float local_sum = 0.0f;

    // TODO 1：
    // 从 idx 开始，每次前进 step，累加自己负责的输入
    for (int i = idx; i < N; i += step) {
        // ...
        local_sum += input[i];
    }

    // TODO 2：把当前线程的局部和写入共享内存
    shared[tid] = local_sum;

    __syncthreads();

    // 将 block 内 256 个线程的局部和合并
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    // TODO 3：由线程 0 写出当前 block 的部分和
    // ...
    if(tid == 0)
    {
        partial[blockIdx.x] = shared[0];
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }

    return value;
}


__global__ void sum_blocks_shuffle(const float* input, float* partial, int N)
{
    // 固定 256 个线程，每个 warp 输出一个部分和
    __shared__ float warp_sums[8];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int step = blockDim.x * gridDim.x;

    // TODO 1：计算 warp 内编号和 block 内的 warp 编号
    int lane = tid % 32;
    int warp_id = tid / 32;

    // 第一阶段：每个线程独立累加多个输入
    float local_sum = 0.0f;

    for (int i = idx; i < N; i += step) {
        local_sum += input[i];
    }

    // 第二阶段：每个 warp 内归约
    // 所有线程都调用，包含没有读到输入的线程
    local_sum = warp_reduce_sum(local_sum);

    // TODO 2：
    // 每个 warp 仅由 lane 0 将总和写到 warp_sums[warp_id]
    // ...
    if(lane == 0)
    {
        warp_sums[warp_id] = local_sum;
    }

    // TODO 3：
    // 等待全部 8 个 warp 写完共享内存
    // 注意：必须让整个 block 执行
    // ...
    __syncthreads();

    // 第三阶段：第一个 warp 合并 8 个部分和
    if (warp_id == 0) {
        // TODO 4：
        // lane 0～7 读取 warp_sums[lane]
        // lane 8～31 使用 0.0f
        float block_sum = (lane < 8) ? warp_sums[lane] : 0.0f;

        // 第一个 warp 的全部 32 个线程都参与
        block_sum = warp_reduce_sum(block_sum);

        // TODO 5：
        // 仅 lane 0 将 block_sum 写入 partial[blockIdx.x]
        // ...
        if(lane == 0)
        {
            partial[blockIdx.x] = block_sum;
        }
    }
}

int main()
{
    std::ofstream log("reduction_results.txt", std::ios::trunc);

    if (!log.is_open()) {
        std::fprintf(stderr, "Failed to open output file.\n");
        return EXIT_FAILURE;
    }

    log << std::fixed << std::setprecision(6);

    auto check_cuda = [](cudaError_t error) {
        if (error != cudaSuccess) {
            std::fprintf(
                stderr, "CUDA error: %s\n",
                cudaGetErrorString(error)
            );
            std::exit(EXIT_FAILURE);
        }
    };

    const int threads = 256;

    const int test_sizes[] = {
        // 1,
        // 17,
        // 33,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26
    };

    const int block_limits[] = {256, 1024, 4096};
    const char* version_names[] = {"stride", "shuffle"};

    const int warmup = 5;
    const int repeats = 300;
    const int batches = 5;

    int total_tests = 0;
    int failed_tests = 0;

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (int N : test_sizes) {
        // 1. 构造包含正数、负数和零的输入
        std::vector<float> input(N);

        for (int i = 0; i < N; ++i) {
            input[i] = static_cast<float>((i % 17) - 8);
        }

        const double cpu_result = sum_cpu(input.data(), N);

        const size_t input_bytes =
            static_cast<size_t>(N) * sizeof(float);

        // 2. 分配并上传输入，不计入归约耗时
        float* dev_input = nullptr;

        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&dev_input),
            input_bytes
        ));

        check_cuda(cudaMemcpy(
            dev_input,
            input.data(),
            input_bytes,
            cudaMemcpyHostToDevice
        ));

        for (int max_blocks : block_limits) {
            log << "========== N = " << N
                << ", max_blocks = " << max_blocks
                << " ==========\n";

            const int capacity = std::min(
                (N + threads - 1) / threads,
                max_blocks
            );

            const size_t buffer_bytes =
                static_cast<size_t>(capacity) * sizeof(float);

            float* dev_a = nullptr;
            float* dev_b = nullptr;

            check_cuda(cudaMalloc(
                reinterpret_cast<void**>(&dev_a),
                buffer_bytes
            ));

            check_cuda(cudaMalloc(
                reinterpret_cast<void**>(&dev_b),
                buffer_bytes
            ));

            // 下标 0：stride，下标 1：shuffle
            float median_ms[2] = {0.0f, 0.0f};
            bool version_passed[2] = {false, false};

            for (int version = 0; version < 2; ++version) {
                int rounds = 0;

                // 3. 执行一次完整的多轮归约
                auto run_reduction = [&]() -> const float* {
                    // 每次调用都从原始输入重新开始
                    int current_N = N;
                    const float* current_input = dev_input;
                    float* current_output = dev_a;

                    rounds = 0;

                    while (current_N > 1) {
                        const int blocks = std::min(
                            (current_N + threads - 1) / threads,
                            max_blocks
                        );

                        if (version == 0) {
                            sum_blocks_stride<<<blocks, threads>>>(
                                current_input,
                                current_output,
                                current_N
                            );
                        } else {
                            sum_blocks_shuffle<<<blocks, threads>>>(
                                current_input,
                                current_output,
                                current_N
                            );
                        }

                        check_cuda(cudaGetLastError());

                        current_input = current_output;
                        current_N = blocks;

                        current_output =
                            (current_output == dev_a) ? dev_b : dev_a;

                        ++rounds;
                    }

                    return current_input;
                };

                // 4. 先验证正确性
                const float* result_ptr = run_reduction();
                check_cuda(cudaDeviceSynchronize());

                float gpu_result = 0.0f;

                check_cuda(cudaMemcpy(
                    &gpu_result,
                    result_ptr,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ));

                const double error = std::abs(
                    static_cast<double>(gpu_result) - cpu_result
                );

                // 此阈值用于当前小整数输入和测试规模
                const bool passed =
                    std::isfinite(cpu_result) &&
                    std::isfinite(gpu_result) &&
                    error < 1e-6;

                version_passed[version] = passed;

                ++total_tests;

                if (!passed) {
                    ++failed_tests;
                }

                log << "Version:   " << version_names[version] << '\n'
                    << "Rounds:    " << rounds << '\n'
                    << "CPU sum:   " << cpu_result << '\n'
                    << "GPU sum:   " << gpu_result << '\n'
                    << "Abs error: " << std::scientific << error
                    << std::fixed << '\n'
                    << "Result:    " << (passed ? "PASS" : "FAIL")
                    << '\n';

                if (!passed) {
                    log << "Timing skipped: validation failed.\n\n";
                    continue;
                }

                // N == 1 时没有启动 kernel，不进行性能比较
                if (N == 1) {
                    log << "Timing skipped: no kernel launched.\n\n";
                    continue;
                }

                // 5. 预热
                for (int i = 0; i < warmup; ++i) {
                    run_reduction();
                }

                check_cuda(cudaDeviceSynchronize());

                // 6. 每批运行 repeats 次完整归约
                std::vector<float> batch_times;
                batch_times.reserve(batches);

                for (int batch = 0; batch < batches; ++batch) {
                    check_cuda(cudaEventRecord(start));

                    for (int i = 0; i < repeats; ++i) {
                        run_reduction();
                    }

                    check_cuda(cudaEventRecord(stop));
                    check_cuda(cudaEventSynchronize(stop));

                    float total_ms = 0.0f;

                    check_cuda(cudaEventElapsedTime(
                        &total_ms, start, stop
                    ));

                    batch_times.push_back(total_ms / repeats);
                }

                // 所有批次计时结束后再输出，避免文件写入影响计时
                for (int batch = 0; batch < batches; ++batch) {
                    log << "Batch " << batch + 1
                        << " average: " << batch_times[batch]
                        << " ms\n";
                }

                // 7. 取五个批次平均耗时的中位数
                std::sort(batch_times.begin(), batch_times.end());

                median_ms[version] = batch_times[batches / 2];

                log << "Median reduction time: "
                    << median_ms[version] << " ms\n\n";
            }

            // 8. 计算 shuffle 相对于 stride 的加速比
            if (N == 1) {
                log << "Speedup: N/A (no kernel launched)\n";
            } else if (!version_passed[0] || !version_passed[1]) {
                log << "Speedup: N/A (validation failed)\n";
            } else if (median_ms[0] > 0.0f && median_ms[1] > 0.0f) {
                const double speedup =
                    static_cast<double>(median_ms[0]) / median_ms[1];

                log << "Stride median time:  "
                    << median_ms[0] << " ms\n";

                log << "Shuffle median time: "
                    << median_ms[1] << " ms\n";

                log << "Shuffle speedup (stride / shuffle): "
                    << speedup << "x\n";
            } else {
                log << "Speedup: N/A (invalid timing)\n";
            }

            log << '\n';

            check_cuda(cudaFree(dev_a));
            check_cuda(cudaFree(dev_b));
        }

        check_cuda(cudaFree(dev_input));
    }

    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    const bool all_passed = (failed_tests == 0);

    log << "Total tests:  " << total_tests << '\n'
        << "Passed tests: " << total_tests - failed_tests << '\n'
        << "Failed tests: " << failed_tests << '\n'
        << "All tests: " << (all_passed ? "PASS" : "FAIL") << '\n';

    log.close();

    if (!log) {
        std::fprintf(stderr, "Failed to write output file.\n");
        return EXIT_FAILURE;
    }

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}