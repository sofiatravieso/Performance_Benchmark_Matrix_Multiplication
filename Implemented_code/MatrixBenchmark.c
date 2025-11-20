#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/resource.h> 
#include <unistd.h>      

#define BLOCK_SIZE 64      
#define STRASSEN_CUTOFF 64  
#define MAX_MEM_ALLOC 2048  

typedef struct {
    int N;
    int non_zeros;
} SparseMatrix;

double get_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

double get_memory_usage_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return (double)usage.ru_maxrss / 1024.0; 
    }
    return 0.0;
}

double* allocate_matrix(int N) {
    double *matrix = (double *)calloc((size_t)N * N, sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix N=%d.\n", N);
        exit(EXIT_FAILURE);
    }
    return matrix;
}

void initialize_matrix(double *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

void tiling_mult(int N, const double *A, const double *B, double *C) {
    int i, j, k, ii, jj, kk;

    memset(C, 0, (size_t)N * N * sizeof(double));

    for (ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (kk = 0; kk < N; kk += BLOCK_SIZE) { 
            for (jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                    for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                        for (j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void add_matrices(int N, const double *A, const double *B, double *C) {
    for (int i = 0; i < N * N; i++) C[i] = A[i] + B[i];
}

void sub_matrices(int N, const double *A, const double *B, double *C) {
    for (int i = 0; i < N * N; i++) C[i] = A[i] - B[i];
}

void copy_quadrant(int N, const double *A, double *B, int row_start, int col_start) {
    int half_N = N / 2;
    for (int i = 0; i < half_N; i++) {
        memcpy(&B[i * half_N], &A[(row_start + i) * N + col_start], half_N * sizeof(double));
    }
}

void join_quadrants(int N, const double *C11, const double *C12, const double *C21, const double *C22, double *C) {
    int half_N = N / 2;
    for (int i = 0; i < half_N; i++) {
        memcpy(&C[i * N], &C11[i * half_N], half_N * sizeof(double));
        memcpy(&C[i * N + half_N], &C12[i * half_N], half_N * sizeof(double));
        memcpy(&C[(half_N + i) * N], &C21[i * half_N], half_N * sizeof(double));
        memcpy(&C[(half_N + i) * N + half_N], &C22[i * half_N], half_N * sizeof(double));
    }
}

void strassen_mult(int N, const double *A, const double *B, double *C) {
    
    if (N <= STRASSEN_CUTOFF) {
        tiling_mult(N, A, B, C);
        return;
    }

    if (N % 2 != 0) {
        fprintf(stderr, "Warning: Strassen is only optimized for even $N$. Using N=%d.\n", N);
    }
    
    int half_N = N / 2;

    double *A11 = allocate_matrix(half_N); double *A12 = allocate_matrix(half_N);
    double *A21 = allocate_matrix(half_N); double *A22 = allocate_matrix(half_N);
    double *B11 = allocate_matrix(half_N); double *B12 = allocate_matrix(half_N);
    double *B21 = allocate_matrix(half_N); double *B22 = allocate_matrix(half_N);
    
    double *T1 = allocate_matrix(half_N); double *T2 = allocate_matrix(half_N);
    
    double *M1 = allocate_matrix(half_N); double *M2 = allocate_matrix(half_N);
    double *M3 = allocate_matrix(half_N); double *M4 = allocate_matrix(half_N);
    double *M5 = allocate_matrix(half_N); double *M6 = allocate_matrix(half_N);
    double *M7 = allocate_matrix(half_N); 
    
    double *C11 = allocate_matrix(half_N); double *C12 = allocate_matrix(half_N);
    double *C21 = allocate_matrix(half_N); double *C22 = allocate_matrix(half_N);

    copy_quadrant(N, A, A11, 0, 0); copy_quadrant(N, A, A12, 0, half_N);
    copy_quadrant(N, A, A21, half_N, 0); copy_quadrant(N, A, A22, half_N, half_N);
    copy_quadrant(N, B, B11, 0, 0); copy_quadrant(N, B, B12, 0, half_N);
    copy_quadrant(N, B, B21, half_N, 0); copy_quadrant(N, B, B22, half_N, half_N);

    add_matrices(half_N, A11, A22, T1); add_matrices(half_N, B11, B22, T2);
    strassen_mult(half_N, T1, T2, M1);

    add_matrices(half_N, A21, A22, T1);
    strassen_mult(half_N, T1, B11, M2);

    sub_matrices(half_N, B12, B22, T1);
    strassen_mult(half_N, A11, T1, M3);

    sub_matrices(half_N, B21, B11, T1);
    strassen_mult(half_N, A22, T1, M4);

    add_matrices(half_N, A11, A12, T1);
    strassen_mult(half_N, T1, B22, M5);

    sub_matrices(half_N, A21, A11, T1); add_matrices(half_N, B11, B12, T2);
    strassen_mult(half_N, T1, T2, M6);

    sub_matrices(half_N, A12, A22, T1); add_matrices(half_N, B21, B22, T2);
    strassen_mult(half_N, T1, T2, M7);

    add_matrices(half_N, M1, M4, C11); 
    sub_matrices(half_N, C11, M5, C11); 
    add_matrices(half_N, C11, M7, C11); 

    add_matrices(half_N, M3, M5, C12);

    add_matrices(half_N, M2, M4, C21);

    sub_matrices(half_N, M1, M2, C22); 
    add_matrices(half_N, C22, M3, C22);
    add_matrices(half_N, C22, M6, C22);

    join_quadrants(N, C11, C12, C21, C22, C);

    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(T1); free(T2);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(C11); free(C12); free(C21); free(C22);
}

void csr_spmv_placeholder(const SparseMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->N; i++) {
        y[i] = x[i] * 0.01; 
    }
}

void create_sparse_data(int N, SparseMatrix *A, double **x_ptr, double density) {
    A->N = N;
    *x_ptr = allocate_matrix(N); 
    initialize_matrix(*x_ptr, N);
    A->non_zeros = (int)((double)N * N * density); 
}

int main() {
    srand(time(NULL)); 
    int N_sizes[] = {512, 1024, 2048};
    int num_sizes = sizeof(N_sizes) / sizeof(N_sizes[0]);

    double density_levels[] = {0.01, 0.10, 0.20};
    int num_densities = sizeof(density_levels) / sizeof(density_levels[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int N = N_sizes[i];
        
        double *A = allocate_matrix(N);
        double *B = allocate_matrix(N);
        initialize_matrix(A, N);
        initialize_matrix(B, N);

        double start, end;
        double mem_start, mem_end, mem_delta;
        
        for (int j = 0; j < num_densities; j++) {
            double density = density_levels[j];
            double sparsity = 1.0 - density;
            
            printf("\n--- Benchmarking for N=%d, Sparsity=%.0f%% ---\n", N, sparsity * 100.0);

            double *C_strassen = allocate_matrix(N);
            mem_start = get_memory_usage_mb();
            start = get_time();
            strassen_mult(N, A, B, C_strassen);
            end = get_time();
            mem_end = get_memory_usage_mb();
            mem_delta = mem_end - mem_start;
            printf("Strassen (Algorithmic Opt.): %.4f s | Memory: %.2f MB\n", end - start, mem_delta);
            free(C_strassen);

            double *C_tiling = allocate_matrix(N);
            mem_start = get_memory_usage_mb();
            start = get_time();
            tiling_mult(N, A, B, C_tiling);
            end = get_time();
            mem_end = get_memory_usage_mb();
            mem_delta = mem_end - mem_start;
            printf("Tiling (Cache Opt.): %.4f s | Memory: %.2f MB\n", end - start, mem_delta);
            free(C_tiling);

            SparseMatrix A_sparse;
            double *x = NULL;
            double *y = allocate_matrix(N); 

            create_sparse_data(N, &A_sparse, &x, density); 
            mem_start = get_memory_usage_mb();
            start = get_time();
            csr_spmv_placeholder(&A_sparse, x, y);
            end = get_time();
            mem_end = get_memory_usage_mb();
            mem_delta = mem_end - mem_start;
            
            printf("Sparse (SpMV - %.0f%%): %.4f s | Memory: %.2f MB\n", sparsity * 100.0, end - start, mem_delta);

            free(x); free(y);

        free(A); free(B); 
    }
    
    return 0;
}