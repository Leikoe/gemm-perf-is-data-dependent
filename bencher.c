#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <stdint.h>

union DoubleBits {
    double d;
    uint64_t u;
};

// stolen from https://stackoverflow.com/questions/68804469/subtract-two-timespec-objects-find-difference-in-time-or-duration
struct timespec diff_timespec(const struct timespec *time0, const struct timespec *time1) {
    assert(time0);
    assert(time1);
    struct timespec diff = {.tv_sec = time1->tv_sec - time0->tv_sec,
                            .tv_nsec = time1->tv_nsec - time0->tv_nsec};
    if (diff.tv_nsec < 0) {
        diff.tv_nsec += 1000000000; // nsec/sec
        diff.tv_sec--;
    }
    return diff;
}

double *alloc_matrix(const int size);
double *alloc_const_matrix(const int size, const int c);
double *alloc_interval_matrix(const int size);
double *alloc_random_matrix(const int size, const int n_bits_to_mask);

double *alloc_matrix(const int size) {
    double *mat = (double *)calloc(size * size, sizeof(double));
    assert(mat);
    return mat;
}

double *alloc_const_matrix(const int size, const int c) {
    double *mat = alloc_matrix(size);
    for (int i = 0; i < size * size; i++) {
        mat[i] = c;
    }
    return mat;
}

double *alloc_interval_matrix(const int size) {
    double *mat = alloc_matrix(size);
    for (int i = 0; i < size * size; i++) {
        mat[i] = (double)i / (size * size - 1);
    }
    return mat;
}

double *alloc_random_matrix(const int size, const int n_bits_to_mask) {
    double *mat = alloc_matrix(size);
    
    uint64_t mask = (~0ULL) << n_bits_to_mask; 

    for (int i = 0; i < size * size; i++) {
        double rnd_val = (double)rand() / RAND_MAX;
        
        union DoubleBits converter;
        converter.d = rnd_val;
        converter.u &= mask;
        mat[i] = converter.d;
    }
    return mat;
}

#define MAT_SIZE 2048
#define NB_REPET 5

struct timespec get_duration_random(int mask_size) {
    double *A, *B, *C;
    A = alloc_random_matrix(MAT_SIZE, mask_size); 
    B = alloc_random_matrix(MAT_SIZE, mask_size);
    C = alloc_const_matrix(MAT_SIZE, 0);
    
    struct timespec start, end;
    assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                MAT_SIZE, MAT_SIZE, MAT_SIZE, 
                1.0, A, MAT_SIZE, B, MAT_SIZE, 
                1.0, C, MAT_SIZE);
                
    assert(clock_gettime(CLOCK_MONOTONIC, &end) == 0);
    
    free(A);
    free(B);
    free(C);
    return diff_timespec(&start, &end);
}

void print_progress(int current, int total) {
    float percentage = (float)current / total * 100.0;
    printf("\rProgress: %.1f%% (%d/%d)", percentage, current, total);
    fflush(stdout);
}

double to_gflops(struct timespec t) {
    double time_sec = (double)t.tv_sec + (double)t.tv_nsec / 1e9;
    double total_ops = 2.0 * (double)MAT_SIZE * (double)MAT_SIZE * (double)MAT_SIZE;
    return (total_ops / time_sec) / 1e9;
}

int main(void) {
    putenv("OMP_NUM_THREADS=1"); 

    FILE *fp = fopen("benchmark_results.csv", "w");
    if (fp == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    fprintf(fp, "n_bits_zeroed,gflops\n");
    
    struct timespec cur;
    
    int steps_per_repet = 27; 
    int total_steps = NB_REPET * steps_per_repet;
    int current_step = 0;

    printf("Starting benchmark (Results -> benchmark_results.csv)...\n");

    printf("Warming up...\n");
    get_duration_random(0);
    printf("Warm up done.\n");

    for (int i = 0; i < NB_REPET; i++) {
        for (int j = 0; j <= 53; j += 2) {
            cur = get_duration_random(j);
            fprintf(fp, "%d,%.6f\n", j, to_gflops(cur));
            print_progress(++current_step, total_steps);
        } 
    }

    printf("\nDone.\n");
    fclose(fp);
    return EXIT_SUCCESS;
}
