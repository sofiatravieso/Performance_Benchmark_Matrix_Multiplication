import numpy as np
import scipy.sparse as sps
import time
import sys
import psutil
import os

sys.setrecursionlimit(5000) 

def strassen_matrix_mult(A, B, cutoff=64):
    N = A.shape[0]
    if N <= cutoff:
        return np.dot(A, B)
    half_N = N // 2
    
    A11 = A[:half_N, :half_N]; A12 = A[:half_N, half_N:]
    A21 = A[half_N:, :half_N]; A22 = A[half_N:, half_N:]
    B11 = B[:half_N, :half_N]; B12 = B[:half_N, half_N:]
    B21 = B[half_N:, :half_N]; B22 = B[half_N:, half_N:]

    M1 = strassen_matrix_mult(A11 + A22, B11 + B22, cutoff)
    M2 = strassen_matrix_mult(A21 + A22, B11, cutoff)
    M3 = strassen_matrix_mult(A11, B12 - B22, cutoff)
    M4 = strassen_matrix_mult(A22, B21 - B11, cutoff)
    M5 = strassen_matrix_mult(A11 + A12, B22, cutoff)
    M6 = strassen_matrix_mult(A21 - A11, B11 + B12, cutoff)
    M7 = strassen_matrix_mult(A12 - A22, B21 + B22, cutoff)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    C = np.zeros((N, N), dtype=A.dtype)
    C[:half_N, :half_N] = C11
    C[:half_N, half_N:] = C12
    C[half_N:, :half_N] = C21
    C[half_N:, half_N:] = C22
    return C

def tiling_matrix_mult(A, B, block_size=64):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=A.dtype)
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            for k in range(0, N, block_size):
                i_end = min(i + block_size, N); j_end = min(j + block_size, N); k_end = min(k + block_size, N)
                C[i:i_end, j:j_end] += np.dot(A[i:i_end, k:k_end], B[k:k_end, j:j_end])
    return C

def sparse_matrix_mult(A_sparse, B_sparse):
    C_sparse = A_sparse @ B_sparse
    return C_sparse

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark(N, sparsity_level):
    density = 1 - sparsity_level
    print(f"\n--- Benchmarking N={N}, Sparsity={sparsity_level*100:.0f}% ---")
    
    A_dense = np.random.rand(N, N)
    B_dense = np.random.rand(N, N)
    
    A_sparse = sps.rand(N, N, density=density, format="csr")
    B_sparse = sps.rand(N, N, density=density, format="csr")

    mem_start = get_memory_usage_mb()
    start_time = time.perf_counter()
    strassen_matrix_mult(A_dense, B_dense)
    end_time = time.perf_counter()
    mem_end = get_memory_usage_mb()
    print(f"Strassen (Algorithmic Opt.): {end_time - start_time:.4f} s | Memory: {mem_end - mem_start:.2f} MB")

    mem_start = get_memory_usage_mb()
    start_time = time.perf_counter()
    tiling_matrix_mult(A_dense, B_dense, block_size=64)
    end_time = time.perf_counter()
    mem_end = get_memory_usage_mb()
    print(f"Tiling (Cache Opt.): {end_time - start_time:.4f} s | Memory: {mem_end - mem_start:.2f} MB")
    
    mem_start = get_memory_usage_mb()
    start_time = time.perf_counter()
    sparse_matrix_mult(A_sparse, B_sparse)
    end_time = time.perf_counter()
    mem_end = get_memory_usage_mb()
    print(f"Sparse (Sparsity {sparsity_level*100:.0f}%): {end_time - start_time:.4f} s | Memory: {mem_end - mem_start:.2f} MB")

if __name__ == "__main__":
    matrix_sizes = [512, 1024, 2048]
    sparsity_levels = [0.99, 0.90, 0.80]
    
    for N in matrix_sizes:
        for sparsity in sparsity_levels:
            run_benchmark(N, sparsity_level=sparsity)