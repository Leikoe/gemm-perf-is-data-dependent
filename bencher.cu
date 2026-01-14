#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#define ITERATIONS 20
#define SIZE 16384

// --- Error Handling Macros ---
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(func) { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error\n"); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CURAND(func) { \
    curandStatus_t status = (func); \
    if (status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND Error\n"); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Kernels ---

// Helper to clear L2 cache
__global__ void flush_l2_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        data[i] = 0.0f;
    }
}

// Kernel to zero out the N Least Significant Bits (LSB)
__global__ void mask_bits_kernel(float* data, int n_bits, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Create mask: Keep upper bits, zero lower n_bits
    // e.g., n_bits=2 -> ...11111100
    unsigned int mask = (n_bits >= 32) ? 0 : (0xFFFFFFFFu << n_bits);

    for (size_t i = idx; i < n; i += stride) {
        unsigned int raw = __float_as_uint(data[i]);
        raw = raw & mask;
        data[i] = __uint_as_float(raw);
    }
}

// --- Benchmark Function ---

void run_benchmark_step(cublasHandle_t handle,
                        float *d_A, float *d_B, float *d_C,
                        float *d_flush, size_t flush_count,
                        cudaEvent_t start, cudaEvent_t stop,
                        int M, int N, int K,
                        int n_bits_zeroed)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops_conversion = 1.0e12;

    int threads = 256;
    int blocks = (flush_count + threads - 1) / threads;

    double total_ms = 0.0;
    double total_tflops = 0.0;

    // Warmup
    for (int i = 0; i < ITERATIONS; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_A, K, d_B, K, &beta, d_C, M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < ITERATIONS; ++i) {
        // Flush L2
        flush_l2_kernel<<<blocks, threads>>>(d_flush, flush_count);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_A, K, d_B, K, &beta, d_C, M));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        double seconds = milliseconds / 1000.0;
        double current_tflops = (total_flops / seconds) / tflops_conversion;

        total_ms += milliseconds;
        total_tflops += current_tflops;
    }

    double avg_ms = total_ms / ITERATIONS;
    double avg_tflops = total_tflops / ITERATIONS;

    // CSV Output: BitsZeroed, AvgTimeMs, AvgTFLOPS
    printf("%d,%.4f,%.4f\n", n_bits_zeroed, avg_ms, avg_tflops);
}

int main() {
    int M = SIZE;
    int N = SIZE;
    int K = SIZE;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // L2 Cache Info
    int dev_id = 0;
    int l2_bytes = 0;
    CHECK_CUDA(cudaGetDevice(&dev_id));
    CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev_id));
    size_t flush_count = (l2_bytes / sizeof(float));
    float *d_flush;
    CHECK_CUDA(cudaMalloc((void**)&d_flush, l2_bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);
    size_t elements_A = size_A / sizeof(float);
    size_t elements_B = size_B / sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size_C));

    // Generate Initial Random Data
    CHECK_CURAND(curandGenerateUniform(gen, d_A, elements_A));
    CHECK_CURAND(curandGenerateUniform(gen, d_B, elements_B));
    CHECK_CUDA(cudaDeviceSynchronize());

    int threads = 256;
    int blocks_A = (elements_A + threads - 1) / threads;
    int blocks_B = (elements_B + threads - 1) / threads;

    // --- CSV Header ---
    printf("BitsZeroed,AvgTimeMs,AvgTFLOPS\n");

    // Loop 0 to 32 zeroed bits
    for (int n_bits = 0; n_bits <= (8*sizeof(float)); n_bits++) {
        // Apply mask incrementally (no need to reset data)
        if (n_bits > 0) {
            mask_bits_kernel<<<blocks_A, threads>>>(d_A, n_bits, elements_A);
            mask_bits_kernel<<<blocks_B, threads>>>(d_B, n_bits, elements_B);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        run_benchmark_step(handle, d_A, d_B, d_C, d_flush, flush_count, start, stop, M, N, K, n_bits);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_flush));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
