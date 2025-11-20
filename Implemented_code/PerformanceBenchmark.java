import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;

public class SparsityBenchmark {

    private static final int STRASSEN_CUTOFF = 64; 

    public static double[][] add(double[][] A, double[][] B, int N) {
        double[][] C = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        return C;
    }

    public static double[][] subtract(double[][] A, double[][] B, int N) {
        double[][] C = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    public static double[][] strassenMult(double[][] A, double[][] B, int N) {
        if (N <= STRASSEN_CUTOFF) {
            double[][] C = new double[N][N];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < N; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return C;
        }

        int halfN = N / 2;

        double[][] A11 = new double[halfN][halfN]; double[][] A12 = new double[halfN][halfN];
        double[][] A21 = new double[halfN][halfN]; double[][] A22 = new double[halfN][halfN];
        double[][] B11 = new double[halfN][halfN]; double[][] B12 = new double[halfN][halfN];
        double[][] B21 = new double[halfN][halfN]; double[][] B22 = new double[halfN][halfN];

        for (int i = 0; i < halfN; i++) {
            for (int j = 0; j < halfN; j++) {
                A11[i][j] = A[i][j]; A12[i][j] = A[i][j + halfN];
                A21[i][j] = A[i + halfN][j]; A22[i][j] = A[i + halfN][j + halfN];
                B11[i][j] = B[i][j]; B12[i][j] = B[i][j + halfN];
                B21[i][j] = B[i + halfN][j]; B22[i][j] = B[i + halfN][j + halfN];
            }
        }

        double[][] M1 = strassenMult(add(A11, A22, halfN), add(B11, B22, halfN), halfN);
        double[][] M2 = strassenMult(add(A21, A22, halfN), B11, halfN);
        double[][] M3 = strassenMult(A11, subtract(B12, B22, halfN), halfN);
        double[][] M4 = strassenMult(A22, subtract(B21, B11, halfN), halfN);
        double[][] M5 = strassenMult(add(A11, A12, halfN), B22, halfN);
        double[][] M6 = strassenMult(subtract(A21, A11, halfN), add(B11, B12, halfN), halfN);
        double[][] M7 = strassenMult(subtract(A12, A22, halfN), add(B21, B22, halfN), halfN);

        double[][] C11 = add(subtract(add(M1, M4, halfN), M5, halfN), M7, halfN);
        double[][] C12 = add(M3, M5, halfN);
        double[][] C21 = add(M2, M4, halfN);
        double[][] C22 = add(subtract(add(M1, M3, halfN), M2, halfN), M6, halfN);

        double[][] C = new double[N][N];
        for (int i = 0; i < halfN; i++) {
            for (int j = 0; j < halfN; j++) {
                C[i][j] = C11[i][j]; C[i][j + halfN] = C12[i][j];
                C[i + halfN][j] = C21[i][j]; C[i + halfN][j + halfN] = C22[i][j];
            }
        }
        return C;
    }


    private static final int BLOCK_SIZE = 64;

    public static double[][] tilingMult(double[][] A, double[][] B, int N) {
        double[][] C = new double[N][N];
        
        for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                    for (int i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                        for (int k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                            for (int j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
        }
        return C;
    }

    static class SparseMatrix {
        int N;
        
        @SuppressWarnings("unchecked") 
        ArrayList<Entry>[] rows;

        private static class Entry {
            int col;
            double val;
            public Entry(int col, double val) {
                this.col = col;
                this.val = val;
            }
        }

        @SuppressWarnings("unchecked")
        public SparseMatrix(int N) {
            this.N = N;
            this.rows = (ArrayList<Entry>[]) new ArrayList[N]; 
            for (int i = 0; i < N; i++) {
                rows[i] = new ArrayList<>();
            }
        }
        
        @SuppressWarnings("unchecked")
        public SparseMatrix(double[][] D, double sparsityThreshold) {
            this.N = D.length;
            this.rows = (ArrayList<Entry>[]) new ArrayList[N];
            for (int i = 0; i < N; i++) {
                rows[i] = new ArrayList<>();
                for (int j = 0; j < N; j++) {
                    Random rand = new Random();
                    if (rand.nextDouble() < (1.0 - sparsityThreshold)) { 
                        rows[i].add(new Entry(j, D[i][j]));
                    }
                }
            }
        }
    }

    public static double[] sparseMultVector(SparseMatrix A, double[] x) {
        int N = A.N;
        double[] y = new double[N];
        
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (SparseMatrix.Entry entry : A.rows[i]) {
                sum += entry.val * x[entry.col];
            }
            y[i] = sum;
        }
        return y;
    }

    public static double[][] initMatrix(int N) {
        Random rand = new Random();
        double[][] matrix = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = rand.nextDouble();
            }
        }
        return matrix;
    }

    public static double[] initVector(int N) {
        double[] x = new double[N];
        Arrays.fill(x, 1.0); 
        return x;
    }

    public static void runBenchmark(int N, double sparsityThreshold) {
        System.out.println("\n--- Benchmarking for N=" + N + ", Sparsity=" + String.format("%.0f", sparsityThreshold * 100) + "% ---");
        double[][] A_dense = initMatrix(N);
        double[][] B_dense = initMatrix(N);
        
        long startTime, endTime;
        Runtime runtime = Runtime.getRuntime();

        runtime.gc(); 
        long memStart = runtime.totalMemory() - runtime.freeMemory();
        startTime = System.nanoTime();
        strassenMult(A_dense, B_dense, N); 
        endTime = System.nanoTime();
        long memEnd = runtime.totalMemory() - runtime.freeMemory();
        double memUsedMB = (memEnd - memStart) / (1024.0 * 1024.0);
        System.out.printf("Strassen (Algorithmic Opt.): %.4f s | Memory: %.2f MB\n", (endTime - startTime) / 1e9, memUsedMB);

        runtime.gc();
        memStart = runtime.totalMemory() - runtime.freeMemory();
        startTime = System.nanoTime();
        tilingMult(A_dense, B_dense, N);
        endTime = System.nanoTime();
        memEnd = runtime.totalMemory() - runtime.freeMemory();
        memUsedMB = (memEnd - memStart) / (1024.0 * 1024.0);
        System.out.printf("Tiling (Cache Opt.): %.4f s | Memory: %.2f MB\n", (endTime - startTime) / 1e9, memUsedMB);

        double[] x = initVector(N);
        SparseMatrix A_sparse = new SparseMatrix(A_dense, sparsityThreshold); 
        
        runtime.gc();
        memStart = runtime.totalMemory() - runtime.freeMemory();
        startTime = System.nanoTime();
        sparseMultVector(A_sparse, x);
        endTime = System.nanoTime();
        memEnd = runtime.totalMemory() - runtime.freeMemory();
        memUsedMB = (memEnd - memStart) / (1024.0 * 1024.0);
        System.out.printf("Sparse (SpMV - %.0f%%): %.4f s | Memory: %.2f MB\n", sparsityThreshold * 100, (endTime - startTime) / 1e9, memUsedMB);
    }
    
    public static void main(String[] args) {
        int[] matrixSizes = {512, 1024, 2048};
        double[] sparsityLevels = {0.99, 0.90, 0.80};
        
        for (int N : matrixSizes) {
            for (double sparsity : sparsityLevels) {
                runBenchmark(N, sparsity);
            }
        }
    }
}