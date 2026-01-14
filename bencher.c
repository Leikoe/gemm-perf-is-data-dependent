#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <string.h> // added for memset

#define MAT_SIZE 2048
#define NB_REPEAT 10
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
double *alloc_random_matrix(const int size);

double *alloc_matrix(const int size) {
  double *mat = (double *)calloc(size * size, sizeof(double));
  assert(mat);
  return mat;
}

double *alloc_random_matrix(const int size) {
  double *mat = (double*)malloc(size * size * sizeof(double));

  for (int i = 0; i < size * size; i++) {
    double rnd_val = (double)rand() / RAND_MAX;
    mat[i] = rnd_val;
  }

  return mat;
}

struct timespec bench(double *A, double *B, double *C, int mask_size) {
  struct timespec start, end, diff;
  double total_time_sec = 0.0;

  long mask = (~0UL) << mask_size;
  for (int k = 0; k < MAT_SIZE * MAT_SIZE; k++) {
    ((long *) A)[k] &= mask;
    ((long *) B)[k] &= mask;
  }

  for (int i = 0; i < NB_REPEAT; i++) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
        MAT_SIZE, MAT_SIZE, MAT_SIZE, 
        1.0, A, MAT_SIZE, B, MAT_SIZE, 
        0, C, MAT_SIZE);
  } 

  for (int i = 0; i < NB_REPEAT; i++) {
    memset(C, 0, MAT_SIZE * MAT_SIZE * sizeof(double));

    assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
        MAT_SIZE, MAT_SIZE, MAT_SIZE, 
        1.0, A, MAT_SIZE, B, MAT_SIZE, 
        0, C, MAT_SIZE);

    assert(clock_gettime(CLOCK_MONOTONIC, &end) == 0);

    diff = diff_timespec(&start, &end);
    total_time_sec += (double)diff.tv_sec + (double)diff.tv_nsec / 1e9;
  }

  double avg_sec = total_time_sec / NB_REPEAT;

  struct timespec avg_ts;
  avg_ts.tv_sec = (time_t)avg_sec;
  avg_ts.tv_nsec = (long)((avg_sec - avg_ts.tv_sec) * 1e9);

  return avg_ts;
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

  fprintf(fp, "BitsZeroed,,AvgTFLOPS\n");

  struct timespec cur;

  int total_steps = 64; 
  int current_step = 0;

  double *A, *B, *C;
  A = alloc_random_matrix(MAT_SIZE);
  B = alloc_random_matrix(MAT_SIZE);
  C = alloc_matrix(MAT_SIZE);

  printf("Starting benchmark (Results -> benchmark_results.csv)...\n");
  printf("Matrices: %dx%d | Averaging over %d repetitions per mask\n", MAT_SIZE, MAT_SIZE, NB_REPEAT);

  printf("Warming up...\n");
  bench(A, B, C, 0);
  printf("Warm up done.\n");

  for (int j = 0; j < 64; j += 1) {
    // get_duration_random now returns the average timespec for this mask
    cur = bench(A, B, C, j);
    fprintf(fp, "%d,,%.6f\n", j, to_gflops(cur));
    print_progress(++current_step, total_steps);
  } 
  free(A);
  free(B);
  free(C);

  printf("\nDone.\n");
  fclose(fp);
  return EXIT_SUCCESS;
}
