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

#define MAT_SIZE 4096

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

struct timespec get_duration_random() {
  double *A, *B, *C;
  A = alloc_random_matrix(MAT_SIZE, 0);
  B = alloc_random_matrix(MAT_SIZE, 0);
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

int main(void) {
  printf("type,special,duration\n");
  struct timespec cur;
  cur = get_duration_const(0.);
  printf("const,0,%ld.%ld\n", cur.tv_sec, cur.tv_nsec);
  cur = get_duration_const(.987);
  printf("const,.987,%ld.%ld\n", cur.tv_sec, cur.tv_nsec);
  cur = get_duration_const(1);
  printf("const,1,%ld.%ld\n", cur.tv_sec, cur.tv_nsec);
  cur = get_duration_interval();
  printf("interval,0,%ld.%ld\n", cur.tv_sec, cur.tv_nsec);
  for (int i = 0; i < 53; i++) {
    cur = get_duration_random();
    printf("random,%d,%ld.%ld\n", i, cur.tv_sec, cur.tv_nsec);
  } 
  return EXIT_SUCCESS;
}
