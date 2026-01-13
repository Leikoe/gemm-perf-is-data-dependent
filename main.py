import os
os.environ["OMP_NUM_THREADS"] = "1" # disable omp for the gemms
import numpy as np
import time
from statistics import mean, median

N = 4096
ITERS = 100
FLOP = 2*N*N*N

# from https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py#L299C5-L300C98
l2_cache_size_in_mbs = 256
l2_cache = np.empty(int(l2_cache_size_in_mbs * 2**20 / 4), dtype=np.int32)

def bench_gemm(a: np.ndarray, b: np.ndarray, title=None):
    """
    Prints flops of a@b.T
    """
    gflops = []
    for i in range(ITERS):
        l2_cache[:] = 0
        start = time.monotonic()
        c = a @ b.T
        end = time.monotonic()
        s = end - start
        gflops.append(FLOP/s * 1e-9)
    print(f"{title+': ' if title is not None else ''}avg: {mean(gflops):.2f}GFlop/s, min: {min(gflops):.2f}GFlop/s, max: {max(gflops):.2f}GFlop/s, median: {median(gflops):.2f}GFlop/s")

a = np.zeros((N, N), dtype=np.float32)
b = np.zeros((N, N), dtype=np.float32)
bench_gemm(a, b, title="zero initialized A and B")

a = np.random.rand(N, N).astype(np.float32)
b = np.random.rand(N, N).astype(np.float32)
bench_gemm(a, b, title="rand initialized A and B")


def zero_last_n_bits(arr, n):
    mask = ~np.uint32(0) << np.uint32(n)
    arr.view(np.uint32)[:] &= mask
    return arr

for i in range(31): # for 32 bits
    a = zero_last_n_bits(np.random.rand(N, N).astype(np.float32), i)
    b = zero_last_n_bits(np.random.rand(N, N).astype(np.float32), i)
    bench_gemm(a, b, title=f"rand initialized A and B ({i} last bits zeroed out)")
