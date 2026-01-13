#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <time.h>

// stolen from https://stackoverflow.com/questions/68804469/subtract-two-timespec-objects-find-difference-in-time-or-duration
struct timespec diff_timespec(const struct timespec *time0,
    const struct timespec *time1) {
  assert(time0);
  assert(time1);
  struct timespec diff = {.tv_sec = time1->tv_sec - time0->tv_sec, //
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
double *alloc_random_matrix(const int size, const int mask_size);

double *alloc_matrix(const int size) {
  double *mat = (double *) calloc(size * size, sizeof(double));
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

double *alloc_random_matrix(const int size, const int mask_size) {
  double *mat = alloc_matrix(size);
  long mask = 0xFFFFFFFFFFFFFFFF;  // we create a 64bit word to then flip bits to the mask 1 by 1 from the right
  for (int i = 0; i < mask_size; i++) {
    mask ^= (1 << i);
  }
  
  for (int i = 0; i < size * size; i++) {
    double rnd_val = (double)rand() / RAND_MAX;
    mat[i] = (double)((long)rnd_val & mask);
  }
  return mat;
}

#define MAT_SIZE 2048
#define NB_REPET 5

struct timespec get_duration_const(double c) {
  double *A, *B, *C;
  A = alloc_const_matrix(MAT_SIZE, c);
  B = alloc_const_matrix(MAT_SIZE, c);
  C = alloc_matrix(MAT_SIZE);
  struct timespec start, end;
  assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE, MAT_SIZE, 1, A, MAT_SIZE, B, MAT_SIZE, 1, C, MAT_SIZE);
  assert(clock_gettime(CLOCK_MONOTONIC, &end) == 0);
  free(A);
  free(B);
  free(C);
  return diff_timespec(&start, &end);
}

struct timespec get_duration_interval() {
  double *A, *B, *C;
  A = alloc_interval_matrix(MAT_SIZE);
  B = alloc_interval_matrix(MAT_SIZE);
  C = alloc_const_matrix(MAT_SIZE, 0);
  struct timespec start, end;
  assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE, MAT_SIZE, 1, A, MAT_SIZE, B, MAT_SIZE, 1, C, MAT_SIZE);
  assert(clock_gettime(CLOCK_MONOTONIC, &end) == 0);
  free(A);
  free(B);
  free(C);
  return diff_timespec(&start, &end);
}

struct timespec get_duration_random(int mask_size) {
  double *A, *B, *C;
  A = alloc_random_matrix(MAT_SIZE, mask_size); 
  B = alloc_random_matrix(MAT_SIZE, mask_size);
  C = alloc_const_matrix(MAT_SIZE, 0);
  struct timespec start, end;
  assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE, MAT_SIZE, 1, A, MAT_SIZE, B, MAT_SIZE, 1, C, MAT_SIZE);
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

  //fprintf(fp, "type,special,gflops\n");
  fprintf(fp, "n_bits_zeroed,gflops\n");
  
  struct timespec cur;
  
  int steps_per_repet = /*3 + 1*/ + 27; 
  int total_steps = NB_REPET * steps_per_repet;
  int current_step = 0;

  printf("Starting benchmark (Results -> benchmark_results.csv)...\n");


  //get_duration_const(0.);
  //get_duration_const(.987);
  //get_duration_const(1);
  //get_duration_interval();
  printf("warming up\n");
  for (int i = 0; i < 53; i++) {
    get_duration_random(i);
  }
  printf("warmed\n");
  //get_duration_random(0);
  for (int i = 0; i < NB_REPET; i++) {
    // cur = get_duration_const(0.);
    // fprintf(fp, "const,0,%.6f\n", to_gflops(cur));
    // print_progress(++current_step, total_steps);

    // cur = get_duration_const(.987);
    // fprintf(fp, "const,.987,%.6f\n", to_gflops(cur));
    // print_progress(++current_step, total_steps);

    // cur = get_duration_const(1);
    // fprintf(fp, "const,1,%.6f\n", to_gflops(cur));
    // print_progress(++current_step, total_steps);

    // cur = get_duration_interval();
    // fprintf(fp, "interval,0,%.6f\n", to_gflops(cur));
    // print_progress(++current_step, total_steps);

    for (int j = 0; j <= 53; j += 2) {
      cur = get_duration_random(j);
      //fprintf(fp, "random,%d,%.6f\n", j, to_gflops(cur));
      fprintf(fp, "%d,%.6f\n", j, to_gflops(cur));
      print_progress(++current_step, total_steps);
    } 
  }

  printf("\nDone.\n");
  fclose(fp);
  return EXIT_SUCCESS;
}
